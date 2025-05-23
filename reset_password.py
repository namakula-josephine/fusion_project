"""
Test script to directly modify a user's password in the database
for debugging purposes.
"""

import os
import json
from auth_utils import hash_password

# Constants
USERS_DB_FILE = 'users_db.json'

def reset_user_password(username, new_password):
    """Reset a user's password in the database"""
    print(f"\nResetting password for user: {username}")
    
    # Check if users database exists
    if not os.path.exists(USERS_DB_FILE):
        print(f"Error: Users database file not found at {USERS_DB_FILE}")
        return False
    
    # Load users database
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
        print(f"Loaded users database with {len(users)} users")
    except Exception as e:
        print(f"Error loading users database: {e}")
        return False
    
    # Check if user exists
    if username not in users:
        print(f"Error: User '{username}' not found in database")
        print(f"Available users: {list(users.keys())}")
        return False
    
    # Hash the new password
    hashed_password = hash_password(new_password)
    
    # Update user's password
    old_hash = users[username]['password']
    users[username]['password'] = hashed_password
    
    print(f"Changed password hash:")
    print(f"  Old: {old_hash[:10]}...{old_hash[-10:]}")
    print(f"  New: {hashed_password[:10]}...{hashed_password[-10:]}")
    
    # Save updated users database
    try:
        with open(USERS_DB_FILE, 'w') as f:
            json.dump(users, f)
        print(f"Saved users database with updated password")
        return True
    except Exception as e:
        print(f"Error saving users database: {e}")
        return False

def main():
    username = "testuser"
    new_password = "testpassword"
    
    print(f"Resetting password for user '{username}' to '{new_password}'")
    if reset_user_password(username, new_password):
        print(f"Password successfully reset")
    else:
        print(f"Failed to reset password")

if __name__ == "__main__":
    main()
