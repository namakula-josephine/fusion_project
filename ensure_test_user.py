"""
This script creates a test user in the authentication database.
Use it to ensure there's a known valid user for testing purposes.
"""
from auth_utils import hash_password, save_users_db
import json
import os

# Constants
USERS_DB_FILE = 'users_db.json'

def create_test_user():
    username = "testuser"
    password = "testpassword" 
    email = "test@example.com"
    
    # Check if users database exists and load it
    users = {}
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, 'r') as f:
                users = json.load(f)
            print(f"Loaded existing users database with {len(users)} users")
        except Exception as e:
            print(f"Error loading users database: {e}")
    
    # Create or update the test user
    users[username] = {
        "username": username,
        "password": hash_password(password),
        "email": email
    }
    
    # Save the updated users database
    try:
        with open(USERS_DB_FILE, 'w') as f:
            json.dump(users, f)
        print(f"Saved users database with {len(users)} users")
        print(f"Created test user: {username}")
        return True
    except Exception as e:
        print(f"Error saving users database: {e}")
        return False

if __name__ == "__main__":
    print("Creating test user for authentication testing...")
    if create_test_user():
        print("Test user created successfully!")
        print("Username: testuser")
        print("Password: testpassword")
    else:
        print("Failed to create test user")
