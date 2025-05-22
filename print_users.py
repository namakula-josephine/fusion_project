"""
Script to print the contents of the users database in a more readable format
"""

import json
import os

def print_users_db():
    users_db_file = 'users_db.json'
    
    if not os.path.exists(users_db_file):
        print(f"Users database file not found at {users_db_file}")
        return
    
    try:
        with open(users_db_file, 'r') as f:
            users = json.load(f)
        
        print(f"Total users: {len(users)}")
        print("\nUsers in database:")
        
        for username, user_data in users.items():
            # Create a copy with the password partially redacted
            safe_user_data = user_data.copy()
            if 'password' in safe_user_data:
                pw = safe_user_data['password']
                if len(pw) > 15:
                    safe_user_data['password'] = f"{pw[:10]}...{pw[-5:]}"
                else:
                    safe_user_data['password'] = "***REDACTED***"
            
            # Print user details with some formatting
            print(f"\nUsername: {username}")
            for key, value in safe_user_data.items():
                if key != 'username':  # Already printed above
                    print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error reading users database: {str(e)}")

if __name__ == "__main__":
    print_users_db()
