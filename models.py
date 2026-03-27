from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

db = SQLAlchemy()

SUBSCRIPTION_AMOUNT = 599      # ₹599 per year
SUBSCRIPTION_DAYS   = 365      # Annual subscription

class User(db.Model):
    __tablename__ = 'users'

    id          = db.Column(db.Integer,      primary_key=True)
    username    = db.Column(db.String(80),   unique=True, nullable=False)
    password    = db.Column(db.String(120),  nullable=False)
    first_name  = db.Column(db.String(80))
    last_name   = db.Column(db.String(80))
    email       = db.Column(db.String(120))
    role        = db.Column(db.String(20),   default='patient')
    joined      = db.Column(db.DateTime,     default=datetime.utcnow)

    # ── Medical profile ──
    age         = db.Column(db.String(10))
    gender      = db.Column(db.String(20))
    blood_group = db.Column(db.String(10))
    height      = db.Column(db.String(20))
    weight      = db.Column(db.String(20))
    allergies   = db.Column(db.Text)
    conditions  = db.Column(db.Text)
    medications = db.Column(db.Text)

    # ── Subscription / Payment ──
    paid         = db.Column(db.Boolean,  default=False)   # True = active subscription
    paid_at      = db.Column(db.DateTime, nullable=True)   # When last payment was made
    expires_at   = db.Column(db.DateTime, nullable=True)   # Subscription expiry date
    payment_id   = db.Column(db.String(60), nullable=True) # Razorpay payment ID
    payment_amount = db.Column(db.Integer, nullable=True)  # Amount paid in rupees

    @property
    def subscription_status(self):
        """Returns: 'active', 'expiring_soon', 'expired', 'unpaid'"""
        if self.role == 'doctor':
            return 'admin'
        if not self.paid or not self.expires_at:
            return 'unpaid'
        now = datetime.utcnow()
        if self.expires_at < now:
            return 'expired'
        if (self.expires_at - now).days <= 30:
            return 'expiring_soon'
        return 'active'

    @property
    def days_remaining(self):
        if not self.expires_at:
            return 0
        delta = (self.expires_at - datetime.utcnow()).days
        return max(0, delta)

    def activate_subscription(self, payment_id, amount=SUBSCRIPTION_AMOUNT):
        """Call this after successful Razorpay payment."""
        now = datetime.utcnow()
        # Renew from today OR extend from existing expiry (whichever is later)
        base = self.expires_at if (self.expires_at and self.expires_at > now) else now
        self.paid         = True
        self.paid_at      = now
        self.expires_at   = base + timedelta(days=SUBSCRIPTION_DAYS)
        self.payment_id   = payment_id
        self.payment_amount = amount

    def to_dict(self):
        return {
            "id":           self.id,
            "username":     self.username          or "",
            "first_name":   self.first_name        or "",
            "last_name":    self.last_name         or "",
            "email":        self.email             or "",
            "role":         self.role if self.role else "patient",
            "paid":         self.paid,
            "joined":       self.joined.strftime("%Y-%m-%d") if self.joined else "",
            "paid_at":      self.paid_at.strftime("%Y-%m-%d") if self.paid_at else "",
            "expires_at":   self.expires_at.strftime("%Y-%m-%d") if self.expires_at else "",
            "days_remaining":    self.days_remaining,
            "subscription_status": self.subscription_status,
            "payment_id":   self.payment_id        or "",
            "payment_amount": self.payment_amount  or 0,
            "age":          self.age               or "",
            "gender":       self.gender            or "",
            "blood_group":  self.blood_group       or "",
            "height":       self.height            or "",
            "weight":       self.weight            or "",
            "allergies":    self.allergies         or "",
            "conditions":   self.conditions        or "",
            "medications":  self.medications       or "",
        }


class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'

    id          = db.Column(db.Integer,   primary_key=True)
    record_id   = db.Column(db.String(20), unique=True, nullable=False)
    user_id     = db.Column(db.Integer,   db.ForeignKey('users.id'), nullable=False)
    top_disease = db.Column(db.String(100))
    confidence  = db.Column(db.String(20))
    prob_float  = db.Column(db.Float)
    symptoms    = db.Column(db.JSON)
    results     = db.Column(db.JSON)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('diagnoses', lazy=True))

    def to_dict(self):
        return {
            "id":          self.record_id,
            "user_id":     self.user_id,
            "top_disease": self.top_disease  or "",
            "confidence":  self.confidence   or "",
            "prob_float":  self.prob_float   or 0.0,
            "symptoms":    self.symptoms     or [],
            "results":     self.results      or [],
            "date":        self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }


class Booking(db.Model):
    __tablename__ = 'bookings'

    id         = db.Column(db.Integer,    primary_key=True)
    booking_id = db.Column(db.String(20), unique=True, nullable=False)
    user_id    = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    doctor     = db.Column(db.String(100))
    date       = db.Column(db.String(20))
    time       = db.Column(db.String(20))
    type       = db.Column(db.String(20))
    status     = db.Column(db.String(20), default='confirmed')
    reason     = db.Column(db.Text)
    created_at = db.Column(db.DateTime,  default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('bookings', lazy=True))

    def to_dict(self):
        return {
            "booking_id": self.booking_id,
            "doctor":     self.doctor  or "",
            "date":       self.date    or "",
            "time":       self.time    or "",
            "type":       self.type    or "",
            "status":     self.status  or "confirmed",
            "reason":     self.reason  or "",
        }