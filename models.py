from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize the database instance
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(80))
    last_name = db.Column(db.String(80))
    email = db.Column(db.String(120))
    role = db.Column(db.String(20), default='patient') # 'patient' or 'doctor'
    paid = db.Column(db.Boolean, default=True)
    joined = db.Column(db.DateTime, default=datetime.utcnow)

    # Medical profile fields from the settings page
    age = db.Column(db.String(10))
    gender = db.Column(db.String(20))
    blood_group = db.Column(db.String(10))
    height = db.Column(db.String(20))
    weight = db.Column(db.String(20))
    allergies = db.Column(db.Text)
    conditions = db.Column(db.Text)
    medications = db.Column(db.Text)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "role": self.role,
            "joined": self.joined.strftime("%Y-%m-%d %H:%M:%S") if self.joined else None,
            "age": self.age,
            "gender": self.gender,
            "blood_group": self.blood_group,
            "height": self.height,
            "weight": self.weight,
            "allergies": self.allergies,
            "conditions": self.conditions,
            "medications": self.medications
        }

class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'
    
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.String(20), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    top_disease = db.Column(db.String(100))
    confidence = db.Column(db.String(20))
    prob_float = db.Column(db.Float)
    symptoms = db.Column(db.JSON)
    results = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to User
    user = db.relationship('User', backref=db.backref('diagnoses', lazy=True))

    def to_dict(self):
        return {
            "id": self.record_id,
            "user_id": self.user_id,
            "top_disease": self.top_disease,
            "confidence": self.confidence,
            "prob_float": self.prob_float,
            "symptoms": self.symptoms,
            "results": self.results,
            "date": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else None
        }

class Booking(db.Model):
    __tablename__ = 'bookings'
    
    id = db.Column(db.Integer, primary_key=True)
    booking_id = db.Column(db.String(20), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor = db.Column(db.String(100))
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    type = db.Column(db.String(20))
    status = db.Column(db.String(20), default='confirmed')
    reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to User
    user = db.relationship('User', backref=db.backref('bookings', lazy=True))

    def to_dict(self):
        return {
            "booking_id": self.booking_id,
            "doctor": self.doctor,
            "date": self.date,
            "time": self.time,
            "type": self.type,
            "status": self.status,
            "reason": self.reason
        }