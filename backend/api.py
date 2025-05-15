from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
import bcrypt
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'fallback-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.LargeBinary, nullable=False)
    active_tokens = db.Column(db.JSON, default=list)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email
        }

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in signup request")
            return jsonify({"message": "Invalid request data"}), 400

        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        # Validate required fields
        if not all([name, email, password]):
            logger.error(f"Missing required fields in signup")
            return jsonify({"message": "All fields are required"}), 400

        # Check if user exists
        if User.query.filter_by(email=email).first():
            logger.info(f"Signup attempt with existing email: {email}")
            return jsonify({"message": "User already exists"}), 400

        # Hash password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create user
        new_user = User(name=name, email=email, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        
        logger.info(f"New user created: {email}")
        return jsonify({"message": "User created successfully"}), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in signup: {str(e)}", exc_info=True)
        return jsonify({"message": "Internal server error"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in login request")
            return jsonify({"message": "Invalid request data"}), 400

        email = data.get('email')
        password = data.get('password')

        # Validate required fields
        if not all([email, password]):
            logger.error("Missing email or password in login request")
            return jsonify({"message": "Email and password are required"}), 400

        # Find user
        user = User.query.filter_by(email=email).first()
        if not user:
            logger.info(f"Login attempt with non-existent email: {email}")
            return jsonify({"message": "Invalid credentials"}), 401

        # Check password
        try:
            if not bcrypt.checkpw(password.encode('utf-8'), user.password):
                logger.info(f"Failed login attempt for user: {email}")
                return jsonify({"message": "Invalid credentials"}), 401
        except Exception as e:
            logger.error(f"Error checking password: {str(e)}", exc_info=True)
            return jsonify({"message": "Authentication error"}), 500

        # Create token
        access_token = create_access_token(identity=user.id)
        
        # Store token
        if not user.active_tokens:
            user.active_tokens = []
        user.active_tokens.append(access_token)
        db.session.commit()
        
        logger.info(f"Successful login for user: {email}")
        
        return jsonify({
            "token": access_token,
            "user": user.to_dict()
        })

    except Exception as e:
        logger.error(f"Error in login: {str(e)}", exc_info=True)
        return jsonify({"message": "Internal server error"}), 500

# Protect the analyze-image endpoint
@app.route('/analyze-image', methods=['POST'])
@jwt_required()  # This requires a valid JWT token
def analyze_image():
    pass  # Placeholder for analyze_image code

if __name__ == '__main__':
    app.run(debug=True, port=8000)
