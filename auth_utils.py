"""
Utility script for managing user authentication and persistence.
This file provides functions to save and load user data to/from disk,
ensuring that user registrations persist between server restarts.
"""

import json
import os
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# File to store user data
USERS_DB_FILE = 'users_db.json'

def save_users_db(users):
    """Save users to a JSON file for persistence"""
    try:
        with open(USERS_DB_FILE, 'w') as f:
            serializable_users = {}
            for username, user_data in users.items():
                # Need to convert all values to strings to ensure JSON serializability
                serializable_users[username] = {
                    "username": user_data["username"],
                    "password": user_data["password"],  # Already a string (hashed)
                    "email": user_data["email"]
                }
            json.dump(serializable_users, f)
        print(f"Saved {len(users)} users to {USERS_DB_FILE}")
        return True
    except Exception as e:
        print(f"Error saving users database: {str(e)}")
        return False

def load_users_db():
    """Load users from a JSON file if it exists"""
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, 'r') as f:
                users = json.load(f)
                print(f"Loaded {len(users)} users from {USERS_DB_FILE}")
                return users
        except Exception as e:
            print(f"Error loading users database: {str(e)}")
            return {}
    else:
        print(f"Users database file not found at {USERS_DB_FILE}")
        return {}

def hash_password(password):
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)
