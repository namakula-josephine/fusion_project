import requests
from passlib.context import CryptContext
import json
import os

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def save_users_db(users):
    """Save users to a JSON file for persistence"""
    with open('users_db.json', 'w') as f:
        # Need to convert password hashes to strings for JSON serialization
        serializable_users = {}
        for username, user_data in users.items():
            serializable_users[username] = {
                "username": user_data["username"],
                "password": user_data["password"],
                "email": user_data["email"]
            }
        json.dump(serializable_users, f)
    print(f"Saved {len(users)} users to users_db.json")

def load_users_db():
    """Load users from a JSON file if it exists"""
    if os.path.exists('users_db.json'):
        with open('users_db.json', 'r') as f:
            return json.load(f)
    return {}

def test_register_and_login():
    # Register a test user
    username = "testuser2"
    password = "testpassword2"
    email = "test2@example.com"
    
    # Create a direct entry in the users database file
    users = load_users_db()
    
    # Hash the password
    hashed_password = pwd_context.hash(password)
    
    # Add the user to the database
    users[username] = {
        "username": username,
        "password": hashed_password,
        "email": email
    }
    
    # Save the updated database
    save_users_db(users)
    print(f"Created user {username} directly in the database")
    
    # Now try to login through the API
    login_data = {"username": username, "password": password}
    print(f"Attempting to login with: {login_data}")
    
    try:
        login_response = requests.post(
            "http://localhost:8000/api/login",
            data=login_data
        )
        print(f"Login response status: {login_response.status_code}")
        
        if login_response.ok:
            print("Login successful!")
            print(login_response.json())
        else:
            print("Login failed!")
            print(login_response.text)
    except Exception as e:
        print(f"Error during login request: {str(e)}")

if __name__ == "__main__":
    test_register_and_login()
