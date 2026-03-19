"""
MediPredict AI — Full Flask Backend
====================================
Routes: login, signup, logout, home/dashboard, predict,
        history, reports, settings, hospitals (proxy),
        book-consultation, download-report (PDF)
"""

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_file, make_response)
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from functools import wraps
import json, os, datetime, io, uuid

# ── jinja2 / weasyprint for PDF (install: pip install weasyprint) ──
try:
    from weasyprint import HTML as WeasyHTML
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR  = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "medipredict_secret_key_2024"

# ── Jinja2 custom filter: convert "2024-03-19 14:20" → "MAR" ──
@app.template_filter('format_month')
def format_month(date_str):
    try:
        dt = datetime.datetime.strptime(str(date_str)[:7], "%Y-%m")
        return dt.strftime("%b").upper()
    except Exception:
        return ""

# ── File-based DBs ──
USER_DB     = os.path.join(BASE_DIR, "users.json")
HISTORY_DB  = os.path.join(BASE_DIR, "history.json")

# ── Google Maps API Key ──
# Get a free key at: https://console.cloud.google.com/
# Enable "Maps JavaScript API" + "Places API"
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY_HERE")


# ════════════════════════════════════════════════════════════
#  DB HELPERS
# ════════════════════════════════════════════════════════════

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:   return json.load(f)
        except: return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_users():    return load_json(USER_DB)
def save_users(u):   save_json(USER_DB, u)
def load_history():  return load_json(HISTORY_DB)
def save_history(h): save_json(HISTORY_DB, h)


# ════════════════════════════════════════════════════════════
#  AUTH DECORATOR
# ════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════
#  ML MODEL
# ════════════════════════════════════════════════════════════

csv_path = os.path.join(BASE_DIR, "symp_dataset.csv")
df = pd.read_csv(csv_path)
X  = df.drop(columns=["disease"])
y  = df["disease"]
le = LabelEncoder()
y_enc = le.fit_transform(y)
model = GaussianNB()
model.fit(X, y_enc)
symptoms_list = list(X.columns)

# ── Auto-build category map from actual CSV column names ──
# Uses keyword matching so it works with ANY dataset column names
_CATEGORY_KEYWORDS = {
    'general':     ['fever','chill','shiver','malaise','fatigue','sweat',
                    'weight','dehydrat','temperat','weakness','pain'],
    'digestive':   ['nausea','vomit','stomach','abdomin','indigest','acid',
                    'constip','diarrhea','diarrhoea','appetite','belly',
                    'gastro','bowel','liver','jaundice','yellow','ulcer'],
    'neuro':       ['headache','dizzi','vision','balance','concentrat',
                    'sensor','spin','unsteady','slurred','speech',
                    'convuls','seizure','blur','distort'],
    'skin':        ['itch','rash','eruption','peel','blister','pimple',
                    'blackhead','spot','skin','sore','crust','ooze','silver','dent','nail'],
    'respiratory': ['cough','breath','phlegm','congest','runny','throat',
                    'sinus','chest','mucus','sputum','wheez','sneez','nose'],
}

def _build_category_map(symptoms):
    cat_map = {cat: [] for cat in _CATEGORY_KEYWORDS}
    used = set()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for sym in symptoms:
            if any(kw in sym.lower() for kw in keywords):
                if sym not in cat_map[cat]:
                    cat_map[cat].append(sym)
    return cat_map

category_map = _build_category_map(symptoms_list)


# ════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        if u in users and users[u]["password"] == p:
            session.clear()
            session["user"] = u
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        fn = request.form.get("first_name", "")
        ln = request.form.get("last_name", "")
        em = request.form.get("email", "")
        role = request.form.get("role", "patient")
        if u in users:
            return render_template("signup.html", error="Username already exists.")
        users[u] = {
            "password": p, "paid": True,
            "first_name": fn, "last_name": ln,
            "email": em, "role": role,
            "joined": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        save_users(users)
        session.clear()
        session["user"] = u
        return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ════════════════════════════════════════════════════════════
#  MAIN PAGES
# ════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def home():
    users    = load_users()
    user_data = users.get(session["user"], {})
    history  = load_history()
    user_hist = history.get(session["user"], [])
    # Count unique diseases in the model for the stat card
    disease_count = len(le.classes_)

    return render_template(
        "index.html",
        symptoms      = symptoms_list,
        username      = session["user"],
        user_data     = user_data,
        diag_count    = len(user_hist),
        disease_count = disease_count,
        category_map  = category_map,
        maps_key   = GOOGLE_MAPS_API_KEY
    )


@app.route("/history")
@login_required
def history_page():
    history   = load_history()
    user_hist = history.get(session["user"], [])
    return render_template("history.html", username=session["user"],
                           records=list(reversed(user_hist)))


@app.route("/reports")
@login_required
def reports_page():
    history   = load_history()
    user_hist = history.get(session["user"], [])
    return render_template("reports.html", username=session["user"],
                           records=list(reversed(user_hist)))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    users = load_users()
    user_data = users.get(session["user"], {})
    msg = ""
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            user_data["first_name"] = request.form.get("first_name", "")
            user_data["last_name"]  = request.form.get("last_name", "")
            user_data["email"]      = request.form.get("email", "")
            user_data["age"]        = request.form.get("age", "")
            user_data["gender"]     = request.form.get("gender", "")
            user_data["blood_group"]= request.form.get("blood_group", "")
            msg = "Profile updated successfully!"
        elif action == "password":
            old = request.form.get("old_password")
            new = request.form.get("new_password")
            if user_data.get("password") == old:
                user_data["password"] = new
                msg = "Password changed successfully!"
            else:
                msg = "ERROR: Old password is incorrect."
        users[session["user"]] = user_data
        save_users(users)
    return render_template("settings.html", username=session["user"],
                           user_data=user_data, msg=msg)


@app.route("/hospitals")
@login_required
def hospitals_page():
    return render_template("hospitals.html", username=session["user"],
                           maps_key=GOOGLE_MAPS_API_KEY)


@app.route("/consultation")
@login_required
def consultation_page():
    return render_template("consultation.html", username=session["user"])


# ════════════════════════════════════════════════════════════
#  API ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # ── Build feature vector in EXACT column order the model was trained on ──
    # Use a DataFrame so sklearn doesn't raise the feature-name UserWarning
    feat_dict = {s: [int(data.get(s, 0))] for s in symptoms_list}
    feat_df   = pd.DataFrame(feat_dict, columns=symptoms_list)

    probs = model.predict_proba(feat_df)[0]          # shape: (n_classes,)
    top3  = np.argsort(probs)[-3:][::-1]             # indices of top-3 classes

    res = [
        {
            "disease":    le.inverse_transform([i])[0],
            "probability": f"{probs[i] * 100:.2f}%",
            "prob_float":  round(float(probs[i] * 100), 2)
        }
        for i in top3
    ]

    # ── Collect which symptoms were actually selected (value == 1) ──
    selected_symptoms = [s for s in symptoms_list if data.get(s, 0) == 1]

    # ── Persist to history.json ──
    record = {
        "id":          str(uuid.uuid4())[:8],
        "date":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symptoms":    selected_symptoms,
        "results":     res,
        "top_disease": res[0]["disease"],
        "confidence":  res[0]["probability"]
    }
    history = load_history()
    history.setdefault(session["user"], []).append(record)
    save_history(history)

    return jsonify(res)


@app.route("/api/history")
@login_required
def api_history():
    history   = load_history()
    user_hist = history.get(session["user"], [])
    return jsonify(list(reversed(user_hist)))


@app.route("/api/delete-record/<record_id>", methods=["DELETE"])
@login_required
def delete_record(record_id):
    history   = load_history()
    user_hist = history.get(session["user"], [])
    history[session["user"]] = [r for r in user_hist if r["id"] != record_id]
    save_history(history)
    return jsonify({"status": "deleted"})


@app.route("/download-report/<record_id>")
@login_required
def download_report(record_id):
    history   = load_history()
    user_hist = history.get(session["user"], [])
    record    = next((r for r in user_hist if r["id"] == record_id), None)
    users     = load_users()
    user_data = users.get(session["user"], {})

    if not record:
        return "Record not found", 404

    html_content = render_template("report_template.html",
                                   record=record, user_data=user_data,
                                   username=session["user"])

    if PDF_SUPPORT:
        pdf_bytes = WeasyHTML(string=html_content).write_pdf()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"medipredict_report_{record_id}.pdf"
        )
    else:
        # Fallback: return HTML
        response = make_response(html_content)
        response.headers["Content-Type"] = "text/html"
        response.headers["Content-Disposition"] = \
            f'attachment; filename="report_{record_id}.html"'
        return response


@app.route("/book-consultation", methods=["POST"])
@login_required
def book_consultation():
    data = request.json
    # In production: integrate with a real booking API (Practo, Calendly, etc.)
    booking = {
        "booking_id": str(uuid.uuid4())[:8].upper(),
        "doctor":     data.get("doctor", "General Physician"),
        "date":       data.get("date", ""),
        "time":       data.get("time", ""),
        "type":       data.get("type", "online"),
        "status":     "confirmed",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    # Save booking to user record
    users     = load_users()
    user_data = users.get(session["user"], {})
    user_data.setdefault("bookings", []).append(booking)
    users[session["user"]] = user_data
    save_users(users)
    return jsonify(booking)


# ════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)