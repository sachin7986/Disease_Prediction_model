"""
MediPredict AI — Full Flask Backend  (SQLAlchemy version)
══════════════════════════════════════════════════════════
Dev:  SQLite  — no setup, just run python app.py
Prod: Set DATABASE_URL env var to postgres://... — zero code change
"""

import os, datetime, io, uuid
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_file, make_response)
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from functools import wraps

from models import db, User, Diagnosis, Booking

try:
    from weasyprint import HTML as WeasyHTML
    PDF_SUPPORT = True
except (ImportError, OSError) as e:
    print(f"⚠️ Warning: WeasyPrint OS libraries not found. PDF downloads disabled. ({e})")
    PDF_SUPPORT = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ══════════════════════════════════════════════════════════════
#  DATABASE CONFIG
#  Dev  -> SQLite  (medipredict.db in project folder, auto-created)
#  Prod -> export DATABASE_URL=postgresql://user:pass@host/dbname
# ══════════════════════════════════════════════════════════════
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "medipredict_dev_key_change_in_prod")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(BASE_DIR, 'medipredict.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY_HERE")


# ══════════════════════════════════════════════════════════════
#  JINJA2 FILTERS
# ══════════════════════════════════════════════════════════════
@app.template_filter("format_month")
def format_month(date_str):
    try:
        dt = datetime.datetime.strptime(str(date_str)[:7], "%Y-%m")
        return dt.strftime("%b").upper()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
#  ML MODEL
# ══════════════════════════════════════════════════════════════
csv_path = os.path.join(BASE_DIR, "symp_dataset.csv")
df = pd.read_csv(csv_path)
X  = df.drop(columns=["disease"])
y  = df["disease"]
le = LabelEncoder()
y_enc = le.fit_transform(y)
model = GaussianNB()
model.fit(X, y_enc)
symptoms_list = list(X.columns)
disease_count = int(len(le.classes_))

_CATEGORY_KEYWORDS = {
    "general":     ["fever","chill","shiver","malaise","fatigue","sweat","weight","dehydrat","weakness"],
    "digestive":   ["nausea","vomit","stomach","abdomin","indigest","acid","constip","diarrhea",
                    "diarrhoea","appetite","belly","liver","jaundice","yellow","ulcer"],
    "neuro":       ["headache","dizzi","vision","balance","concentrat","sensor","spin",
                    "unsteady","slurred","speech","blur","distort"],
    "skin":        ["itch","rash","eruption","peel","blister","pimple","blackhead",
                    "spot","skin","sore","crust","ooze","nail"],
    "respiratory": ["cough","breath","phlegm","congest","runny","throat","sinus",
                    "chest","mucus","sputum","wheez","sneez","nose"],
}

def _build_category_map(symptoms):
    cat_map = {cat: [] for cat in _CATEGORY_KEYWORDS}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for sym in symptoms:
            if any(kw in sym.lower() for kw in keywords):
                if sym not in cat_map[cat]:
                    cat_map[cat].append(sym)
    return cat_map

category_map = _build_category_map(symptoms_list)


# ══════════════════════════════════════════════════════════════
#  DB INIT
# ══════════════════════════════════════════════════════════════
def init_db():
    with app.app_context():
        db.create_all()
        # Add new subscription columns to existing DB if they don't exist (SQLite migration)
        try:
            with db.engine.connect() as conn:
                cols = [row[1] for row in conn.execute(db.text("PRAGMA table_info(users)")).fetchall()]
                if "paid_at"       not in cols: conn.execute(db.text("ALTER TABLE users ADD COLUMN paid_at DATETIME"))
                if "expires_at"    not in cols: conn.execute(db.text("ALTER TABLE users ADD COLUMN expires_at DATETIME"))
                if "payment_id"    not in cols: conn.execute(db.text("ALTER TABLE users ADD COLUMN payment_id VARCHAR(60)"))
                if "payment_amount" not in cols: conn.execute(db.text("ALTER TABLE users ADD COLUMN payment_amount INTEGER"))
                conn.commit()
        except Exception as e:
            print(f"Migration note: {e}")
        print("Database ready: medipredict.db")


# ══════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        user = current_user()
        if user and user.role == "patient":
            # Check subscription — allow /payment and /logout through
            from flask import request as _req
            if _req.endpoint not in ("payment_page", "confirm_order", "logout"):
                if not user.paid:
                    return redirect(url_for("payment_page"))
                if user.expires_at and user.expires_at < datetime.datetime.utcnow():
                    # Subscription expired — mark as unpaid and redirect
                    user.paid = False
                    db.session.commit()
                    return redirect(url_for("payment_page"))
        return f(*args, **kwargs)
    return decorated

def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None

def doctor_required(f):
    """Ensure only logged-in users with role='doctor' can access."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        user = current_user()
        if not user or user.role != "doctor":
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u    = request.form.get("username", "").strip()
        p    = request.form.get("password", "")
        user = User.query.filter_by(username=u).first()
        if user and user.password == p:   # use check_password_hash in production
            session.clear()
            session["user_id"]  = user.id
            session["username"] = user.username
            session["role"]     = user.role
            # Role-based routing: doctors go to doctor dashboard
            if user.role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u    = request.form.get("username", "").strip()
        p    = request.form.get("password", "")
        fn   = request.form.get("first_name", "")
        ln   = request.form.get("last_name", "")
        em   = request.form.get("email", "")
        role = request.form.get("role", "patient")

        if User.query.filter_by(username=u).first():
            return render_template("signup.html", error="Username already taken.")

        new_user = User(
            username=u, password=p,
            first_name=fn, last_name=ln,
            email=em, role=role,
            paid=True if role == "doctor" else False,  # doctors skip payment
            joined=datetime.datetime.utcnow()
        )
        db.session.add(new_user)
        db.session.commit()
        session.clear()
        session["user_id"]  = new_user.id
        session["username"] = new_user.username
        session["role"]     = new_user.role
        # Doctors bypass payment, go straight to their dashboard
        if new_user.role == "doctor":
            return redirect(url_for("doctor_dashboard"))
        # Patients must pay before accessing the app
        return redirect(url_for("payment_page"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════
@app.route("/")
@login_required
def home():
    user       = current_user()
    diag_count = Diagnosis.query.filter_by(user_id=user.id).count()
    return render_template(
        "index.html",
        symptoms      = symptoms_list,
        username      = user.username,
        user_data     = user.to_dict(),
        diag_count    = diag_count,
        disease_count = disease_count,
        category_map  = category_map,
        maps_key      = GOOGLE_MAPS_API_KEY,
    )


@app.route("/history")
@login_required
def history_page():
    user    = current_user()
    records = (Diagnosis.query.filter_by(user_id=user.id)
               .order_by(Diagnosis.created_at.desc()).all())
    return render_template("history.html",
                           username=user.username,
                           records=[r.to_dict() for r in records])


@app.route("/reports")
@login_required
def reports_page():
    user    = current_user()
    records = (Diagnosis.query.filter_by(user_id=user.id)
               .order_by(Diagnosis.created_at.desc()).all())
    return render_template("reports.html",
                           username=user.username,
                           role=user.role,
                           records=[r.to_dict() for r in records])


@app.route("/doctor/reports")
@doctor_required
def doctor_reports_page():
    """Doctor-facing reports — shows system-wide diagnoses summary."""
    doc     = current_user()
    records = (Diagnosis.query
               .order_by(Diagnosis.created_at.desc())
               .limit(200).all())
    return render_template("reports.html",
                           username=doc.username,
                           role="doctor",
                           records=[r.to_dict() for r in records])


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    user = current_user()
    msg  = ""
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            user.first_name  = request.form.get("first_name", "")
            user.last_name   = request.form.get("last_name", "")
            user.email       = request.form.get("email", "")
            user.age         = request.form.get("age", "")
            user.gender      = request.form.get("gender", "")
            user.blood_group = request.form.get("blood_group", "")
            user.height      = request.form.get("height", "")
            user.weight      = request.form.get("weight", "")
            user.allergies   = request.form.get("allergies", "")
            user.conditions  = request.form.get("conditions", "")
            user.medications = request.form.get("medications", "")
            db.session.commit()
            msg = "Profile updated successfully!"
        elif action == "password":
            old = request.form.get("old_password", "")
            new = request.form.get("new_password", "")
            if user.password == old:
                user.password = new
                db.session.commit()
                msg = "Password changed successfully!"
            else:
                msg = "ERROR: Old password is incorrect."
    return render_template("settings.html",
                           username=user.username,
                           user_data=user.to_dict(), msg=msg)


@app.route("/hospitals")
@login_required
def hospitals_page():
    user = current_user()
    return render_template("hospitals.html",
                           username=user.username,
                           maps_key=GOOGLE_MAPS_API_KEY)


@app.route("/consultation")
@login_required
def consultation_page():
    user     = current_user()
    bookings = (Booking.query.filter_by(user_id=user.id)
                .order_by(Booking.created_at.desc()).all())
    return render_template("consultation.html",
                           username=user.username,
                           my_bookings=[b.to_dict() for b in bookings])


# ══════════════════════════════════════════════════════════════
#  API — PREDICT
# ══════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    user = current_user()
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    feat_dict = {s: [int(data.get(s, 0))] for s in symptoms_list}
    feat_df   = pd.DataFrame(feat_dict, columns=symptoms_list)
    probs     = model.predict_proba(feat_df)[0]
    top3      = np.argsort(probs)[-3:][::-1]

    results = [
        {
            "disease":     le.inverse_transform([i])[0],
            "probability": f"{probs[i]*100:.2f}%",
            "prob_float":  round(float(probs[i]*100), 2),
        }
        for i in top3
    ]

    selected_symptoms = [s for s in symptoms_list if data.get(s, 0) == 1]

    record = Diagnosis(
        record_id   = str(uuid.uuid4())[:8],
        user_id     = user.id,
        created_at  = datetime.datetime.utcnow(),
        top_disease = results[0]["disease"],
        confidence  = results[0]["probability"],
        prob_float  = results[0]["prob_float"],
    )
    record.symptoms = selected_symptoms
    record.results  = results
    db.session.add(record)
    db.session.commit()

    return jsonify(results)


# ══════════════════════════════════════════════════════════════
#  API — HISTORY CRUD
# ══════════════════════════════════════════════════════════════
@app.route("/api/history")
@login_required
def api_history():
    user    = current_user()
    records = (Diagnosis.query.filter_by(user_id=user.id)
               .order_by(Diagnosis.created_at.desc()).all())
    return jsonify([r.to_dict() for r in records])


@app.route("/api/delete-record/<record_id>", methods=["DELETE"])
@login_required
def delete_record(record_id):
    user   = current_user()
    record = Diagnosis.query.filter_by(record_id=record_id, user_id=user.id).first()
    if record:
        db.session.delete(record)
        db.session.commit()
    return jsonify({"status": "deleted"})


# ══════════════════════════════════════════════════════════════
#  API — DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════════
@app.route("/download-report/<record_id>")
@login_required
def download_report(record_id):
    user   = current_user()
    record = Diagnosis.query.filter_by(record_id=record_id, user_id=user.id).first()
    if not record:
        return "Report not found", 404

    html_content = render_template(
        "report_template.html",
        record    = record.to_dict(),
        user_data = user.to_dict(),
        username  = user.username,
    )

    if PDF_SUPPORT:
        pdf_bytes = WeasyHTML(string=html_content).write_pdf()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"medipredict_report_{record_id}.pdf",
        )
    resp = make_response(html_content)
    resp.headers["Content-Type"] = "text/html"
    resp.headers["Content-Disposition"] = f'attachment; filename="report_{record_id}.html"'
    return resp


# ══════════════════════════════════════════════════════════════
#  API — BOOK CONSULTATION
# ══════════════════════════════════════════════════════════════
@app.route("/book-consultation", methods=["POST"])
@login_required
def book_consultation():
    user = current_user()
    data = request.json or {}
    doctor = data.get("doctor", "General Physician")
    date   = data.get("date", "")
    time   = data.get("time", "")

    # ── Duplicate slot check ──────────────────────────────────────
    existing = Booking.query.filter_by(
        doctor=doctor, date=date, time=time
    ).filter(Booking.status != "cancelled").first()
    if existing:
        return jsonify({
            "error": "slot_taken",
            "message": f"Dr. {doctor.split('.')[-1].strip()} is already booked at {time} on {date}. Please choose a different time slot."
        }), 409
    # ─────────────────────────────────────────────────────────────

    booking = Booking(
        booking_id = str(uuid.uuid4())[:8].upper(),
        user_id    = user.id,
        doctor     = doctor,
        date       = date,
        time       = time,
        type       = data.get("type", "online"),
        reason     = data.get("reason", ""),
        status     = "confirmed",
        created_at = datetime.datetime.utcnow(),
    )
    db.session.add(booking)
    db.session.commit()
    return jsonify(booking.to_dict())


# ══════════════════════════════════════════════════════════════
#  ONE-TIME JSON -> DB MIGRATION
#  POST http://127.0.0.1:5001/api/migrate-json
# ══════════════════════════════════════════════════════════════
@app.route("/api/migrate-json", methods=["POST"])
def migrate_json():
    import json as _json
    migrated_users = migrated_diags = 0

    users_path = os.path.join(BASE_DIR, "users.json")
    if os.path.exists(users_path):
        for username, info in _json.load(open(users_path)).items():
            if not User.query.filter_by(username=username).first():
                db.session.add(User(
                    username=username, password=info.get("password",""),
                    email=info.get("email",""), first_name=info.get("first_name",""),
                    last_name=info.get("last_name",""), role=info.get("role","patient"),
                    paid=info.get("paid", True), age=info.get("age",""),
                    gender=info.get("gender",""), blood_group=info.get("blood_group",""),
                ))
                migrated_users += 1
        db.session.commit()

    history_path = os.path.join(BASE_DIR, "history.json")
    if os.path.exists(history_path):
        for username, records in _json.load(open(history_path)).items():
            user = User.query.filter_by(username=username).first()
            if not user:
                continue
            for r in records:
                if Diagnosis.query.filter_by(record_id=r["id"]).first():
                    continue
                d = Diagnosis(
                    record_id=r["id"], user_id=user.id,
                    top_disease=r.get("top_disease",""),
                    confidence=r.get("confidence",""),
                    prob_float=float(r.get("confidence","0%").replace("%","")),
                )
                try:
                    d.created_at = datetime.datetime.strptime(r["date"], "%Y-%m-%d %H:%M")
                except Exception:
                    d.created_at = datetime.datetime.utcnow()
                d.symptoms = r.get("symptoms", [])
                d.results  = r.get("results",  [])
                db.session.add(d)
                migrated_diags += 1
        db.session.commit()

    return jsonify({"status":"done","users_migrated":migrated_users,
                    "diagnoses_migrated":migrated_diags})


# ══════════════════════════════════════════════════════════════
#  PAYMENT / SUBSCRIPTION ROUTES
# ══════════════════════════════════════════════════════════════

SUBSCRIPTION_PRICE = 599   # ₹599 per year

@app.route("/payment")
@login_required
def payment_page():
    user = current_user()
    # Doctors never pay — redirect them away
    if user.role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    # Already paid and not expired — go to app
    if user.paid and user.expires_at and user.expires_at > datetime.datetime.utcnow():
        return redirect(url_for("home"))
    return render_template(
        "payment.html",
        username   = user.username,
        email      = user.email or "",
        amount     = SUBSCRIPTION_PRICE,
        is_renewal = (user.paid_at is not None),  # True = renewing, False = first time
        expires_at = user.expires_at.strftime("%d %b %Y") if user.expires_at else None,
    )


@app.route("/ConfirmOrder")
@login_required
def confirm_order():
    """Called by Razorpay callback after successful payment."""
    user       = current_user()
    payment_id = request.args.get("payment_id", "demo_" + str(uuid.uuid4())[:8])
    user.activate_subscription(payment_id=payment_id, amount=SUBSCRIPTION_PRICE)
    db.session.commit()
    return "success"


@app.route("/api/subscription-status")
@login_required
def subscription_status():
    user = current_user()
    return jsonify({
        "paid":           user.paid,
        "expires_at":     user.expires_at.strftime("%d %b %Y") if user.expires_at else None,
        "days_remaining": user.days_remaining,
        "status":         user.subscription_status,
    })


# ── Doctor-only: get all subscription data for admin panel ──
@app.route("/api/doctor/subscriptions")
@doctor_required
def doctor_subscriptions():
    from sqlalchemy import func
    patients = User.query.filter_by(role="patient").order_by(User.joined.desc()).all()
    now      = datetime.datetime.utcnow()
    result   = {
        "active":        [],
        "expiring_soon": [],
        "expired":       [],
        "unpaid":        [],
        "total_revenue": 0,
    }
    for p in patients:
        d = p.to_dict()
        status = p.subscription_status
        result[status].append(d)
        if p.payment_amount:
            result["total_revenue"] += p.payment_amount
    result["summary"] = {
        "active":        len(result["active"]),
        "expiring_soon": len(result["expiring_soon"]),
        "expired":       len(result["expired"]),
        "unpaid":        len(result["unpaid"]),
        "total_patients":len(patients),
        "total_revenue": result["total_revenue"],
    }
    return jsonify(result)


# ── Doctor manually renews/waives a subscription ──
@app.route("/api/doctor/grant-access/<int:patient_id>", methods=["POST"])
@doctor_required
def grant_access(patient_id):
    patient = db.session.get(User, patient_id)
    if not patient or patient.role != "patient":
        return jsonify({"error": "Patient not found"}), 404
    days = request.json.get("days", 365)
    patient.activate_subscription(payment_id="doctor_granted", amount=0)
    patient.expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    db.session.commit()
    return jsonify({"status": "ok", "expires_at": patient.expires_at.strftime("%d %b %Y")})


@app.route("/api/doctor/revoke-access/<int:patient_id>", methods=["POST"])
@doctor_required
def revoke_access(patient_id):
    patient = db.session.get(User, patient_id)
    if not patient or patient.role != "patient":
        return jsonify({"error": "Patient not found"}), 404
    patient.paid       = False
    patient.expires_at = datetime.datetime.utcnow()
    db.session.commit()
    return jsonify({"status": "revoked"})


# ══════════════════════════════════════════════════════════════
#  DOCTOR / ADMIN ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/doctor")
@doctor_required
def doctor_dashboard():
    doc = current_user()
    # Aggregate stats for the doctor overview
    total_patients  = User.query.filter_by(role="patient").count()
    total_diagnoses = Diagnosis.query.count()
    total_bookings  = Booking.query.count()

    # Top 5 most diagnosed diseases
    from sqlalchemy import func
    top_diseases = (db.session.query(
            Diagnosis.top_disease,
            func.count(Diagnosis.id).label("count"),
            func.round(func.avg(Diagnosis.prob_float), 1).label("avg_conf")
        )
        .group_by(Diagnosis.top_disease)
        .order_by(func.count(Diagnosis.id).desc())
        .limit(8).all()
    )

    # Recent 5 diagnoses across all patients
    recent_diagnoses = (Diagnosis.query
        .join(User, Diagnosis.user_id == User.id)
        .order_by(Diagnosis.created_at.desc())
        .limit(10).all()
    )

    # Recently joined patients
    recent_patients = (User.query
        .filter_by(role="patient")
        .order_by(User.joined.desc())
        .limit(6).all()
    )

    # Recent 10 bookings across all patients
    recent_bookings = (Booking.query
        .join(User, Booking.user_id == User.id)
        .order_by(Booking.created_at.desc())
        .limit(10).all()
    )
    # Subscription summary for doctor dashboard
    from sqlalchemy import func as _func
    now_dt = datetime.datetime.utcnow()
    sub_active   = User.query.filter(User.role=="patient", User.paid==True,
                     User.expires_at > now_dt).count()
    sub_expiring = User.query.filter(User.role=="patient", User.paid==True,
                     User.expires_at > now_dt,
                     User.expires_at <= now_dt + datetime.timedelta(days=30)).count()
    sub_expired  = User.query.filter(User.role=="patient",
                     (User.paid==False) | (User.expires_at <= now_dt)).count()
    total_revenue= db.session.query(_func.sum(User.payment_amount))                     .filter(User.role=="patient", User.payment_amount != None).scalar() or 0

    return render_template(
        "doctor_dashboard.html",
        doctor_name    = doc.first_name or doc.username,
        username       = doc.username,
        total_patients = total_patients,
        total_diagnoses= total_diagnoses,
        total_bookings = total_bookings,
        top_diseases   = [{"disease": r[0], "count": r[1], "avg_conf": float(r[2] or 0)} for r in top_diseases],
        recent_diagnoses = recent_diagnoses,
        recent_patients  = [p.to_dict() for p in recent_patients],
        recent_bookings  = recent_bookings,
        sub_active   = sub_active,
        sub_expiring = sub_expiring,
        sub_expired  = sub_expired,
        total_revenue = total_revenue,
    )


@app.route("/doctor/patients")
@doctor_required
def doctor_patients():
    doc = current_user()
    # All patients with their diagnosis count and last diagnosis
    from sqlalchemy import func
    patients = (db.session.query(
            User,
            func.count(Diagnosis.id).label("diag_count"),
            func.max(Diagnosis.created_at).label("last_diag"),
            func.max(Diagnosis.top_disease).label("last_disease"),
        )
        .outerjoin(Diagnosis, User.id == Diagnosis.user_id)
        .filter(User.role == "patient")
        .group_by(User.id)
        .order_by(func.count(Diagnosis.id).desc())
        .all()
    )
    patient_data = []
    for p, diag_count, last_diag, last_disease in patients:
        d = p.to_dict()
        d["diag_count"]   = diag_count or 0
        d["last_diag"]    = str(last_diag)[:16] if last_diag else "—"
        d["last_disease"] = last_disease or "—"
        patient_data.append(d)

    return render_template("doctor_patients.html",
                           username=doc.username,
                           patients=patient_data)


@app.route("/doctor/patient/<int:patient_id>")
@doctor_required
def doctor_patient_detail(patient_id):
    doc     = current_user()
    patient = User.query.get_or_404(patient_id)
    if patient.role != "patient":
        return redirect(url_for("doctor_patients"))
    records = (Diagnosis.query
               .filter_by(user_id=patient_id)
               .order_by(Diagnosis.created_at.desc())
               .all())
    bookings = (Booking.query
                .filter_by(user_id=patient_id)
                .order_by(Booking.created_at.desc())
                .all())
    return jsonify({
        "patient":  patient.to_dict(),
        "records":  [r.to_dict() for r in records],
        "bookings": [b.to_dict() for b in bookings],
    })


@app.route("/api/doctor/bookings")
@doctor_required
def doctor_all_bookings():
    """Return all bookings with patient info for doctor dashboard."""
    bookings = (Booking.query
                .join(User, Booking.user_id == User.id)
                .order_by(Booking.created_at.desc())
                .limit(200).all())
    result = []
    for b in bookings:
        d = b.to_dict()
        d["patient_name"]  = b.user.first_name or b.user.username
        d["patient_user"]  = b.user.username
        d["created_at"]    = b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else ""
        result.append(d)
    return jsonify(result)


@app.route("/doctor/sql")
@doctor_required
def doctor_sql():
    doc = current_user()
    return render_template("doctor_sql.html", username=doc.username)

@app.route("/api/doctor/sql", methods=["POST"])
@doctor_required
def doctor_run_sql():
    """Doctor SQL — full read access + blocked write operations."""
    import re as _re, time as _time
    _BLOCKED_DR = _re.compile(
        r"\b(DROP|TRUNCATE|ALTER|CREATE|ATTACH|DETACH|VACUUM|REINDEX)\b",
        _re.IGNORECASE
    )
    data  = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    clean = _re.sub(r"--[^\n]*", "", query)
    clean = _re.sub(r"/\*.*?\*/", "", clean, flags=_re.DOTALL).strip()

    if _BLOCKED_DR.search(clean):
        return jsonify({
            "error": "Schema-altering operations (DROP, ALTER, TRUNCATE, CREATE) are blocked."
        }), 400

    if not _re.search(r"\bLIMIT\b", clean, _re.IGNORECASE):
        query = query.rstrip(";") + " LIMIT 1000;"

    try:
        t0     = _time.time()
        result = db.session.execute(db.text(query))
        cols   = list(result.keys())
        rows   = [list(r) for r in result.fetchall()]
        db.session.commit()  # needed if doctor runs INSERT/UPDATE for notes
        return jsonify({"columns": cols, "rows": rows, "elapsed": round(_time.time()-t0, 4)})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

# ══════════════════════════════════════════════════════════════
#  SQL EXPLORER — page + secure query API
# ══════════════════════════════════════════════════════════════
import re as _re, time as _time

# Blocked keywords — prevent destructive / schema-changing queries
_BLOCKED = _re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|EXPLAIN\s+QUERY\s+PLAN)\b",
    _re.IGNORECASE
)

@app.route("/sql-explorer")
@login_required
def sql_explorer():
    user = current_user()
    return render_template("sql_explorer.html", username=user.username)

@app.route("/api/sql", methods=["POST"])
@login_required
def run_sql():
    """
    Execute a read-only SQL query and return results as JSON.
    Blocks all non-SELECT statements for safety.
    """
    data  = request.json or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Clean the query of comments before checking it
    clean = _re.sub(r"--[^\n]*", "", query)
    clean = _re.sub(r"/\*.*?\*/", "", clean, flags=_re.DOTALL).strip()

    # Must start with SELECT (or WITH for CTEs)
    if not _re.match(r"^(SELECT|WITH)\b", clean, _re.IGNORECASE):
        return jsonify({
            "error": "Only SELECT queries are allowed in the Explorer. Write/schema operations are blocked for safety."
        }), 400

    # Block dangerous keywords even inside SELECT
    if _BLOCKED.search(clean):
        return jsonify({
            "error": "Query contains a blocked keyword. Write/schema operations are not permitted."
        }), 400

    # Enforce row limit — cap at 500 rows to avoid huge result sets
    MAX_ROWS = 500
    if not _re.search(r"\bLIMIT\b", clean, _re.IGNORECASE):
        query = query.rstrip(";") + f" LIMIT {MAX_ROWS};"

    try:
        t0      = _time.time()
        result  = db.session.execute(db.text(query))
        columns = list(result.keys())
        rows    = [list(row) for row in result.fetchall()]
        elapsed = round(_time.time() - t0, 4)
        return jsonify({"columns": columns, "rows": rows, "elapsed": elapsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5001, debug=True)