"""
Direct test of the login API endpoint
"""
import requests

def test_login():
    """Test logging in with our known test user"""
    print("Testing login with known test user...")
    
    # Credentials that should match what's in ensure_test_user.py
    username = "testuser"
    password = "testpassword"
    
    # Make the request
    print(f"Sending login request for {username}...")
    response = requests.post(
        "http://localhost:8000/api/login",
        data={
            "username": username,
            "password": password
        }
    )
    
    # Show results
    print(f"Response status: {response.status_code}")
    
    try:
        data = response.json()
        print(f"Response data: {data}")
        
        if response.ok and "session_id" in data:
            print("\n✅ SUCCESS: Authentication works!")
            print(f"Session ID: {data['session_id']}")
        else:
            print("\n❌ FAILURE: Authentication failed")
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response.text}")

if __name__ == "__main__":
    test_login()
