"""
Generate a test user for the FastAPI backend with proper password hashing
This script creates a test user directly in the users_db.json file
"""

import sys
import os
import json
from auth_utils import hash_password, load_users_db, save_users_db

def create_test_user(username, password, email):
    """Create a test user with the given credentials"""
    
    # Load existing users
    users_db = load_users_db()
    
    # Check if user already exists
    if username in users_db:
        print(f"User '{username}' already exists. Updating password...")
    
    # Hash the password and store the user
    hashed_password = hash_password(password)
    users_db[username] = {
        "username": username,
        "password": hashed_password,
        "email": email
    }
    
    # Save the users database
    save_users_db(users_db)
    
    print(f"User '{username}' created/updated successfully!")
    print(f"Total users in database: {len(users_db)}")

if __name__ == "__main__":
    # Default test user
    default_username = "testuser"
    default_password = "testpassword"
    default_email = "test@example.com"
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = default_username
        
    if len(sys.argv) > 2:
        password = sys.argv[2]
    else:
        password = default_password
        
    if len(sys.argv) > 3:
        email = sys.argv[3]
    else:
        email = default_email
    
    # Create the test user
    create_test_user(username, password, email)
