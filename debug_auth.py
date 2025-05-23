"""
Debug script for testing authentication and password verification
"""

from auth_utils import verify_password, load_users_db
import sys

def test_password_verification(username, password):
    """Test if a password can be successfully verified for a given user"""
    
    print(f"\nTesting authentication for user: {username}")
    
    # Load the users database
    users_db = load_users_db()
    
    # Check if user exists
    if username not in users_db:
        print(f"ERROR: User '{username}' not found in database!")
        print(f"Available users: {list(users_db.keys())}")
        return False
    
    # Get the stored hash
    stored_hash = users_db[username]['password']
    print(f"Found password hash: {stored_hash[:10]}...{stored_hash[-5:]}")
    
    # Test password verification
    is_valid = verify_password(password, stored_hash)
    
    if is_valid:
        print(f"SUCCESS: Password verification passed for '{username}'")
    else:
        print(f"FAIL: Password verification failed for '{username}'")
    
    return is_valid

if __name__ == "__main__":
    # Get username and password from command line arguments
    if len(sys.argv) < 3:
        print("Usage: python debug_auth.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    # Test the authentication
    test_password_verification(username, password)
