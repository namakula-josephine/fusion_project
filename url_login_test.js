/**
 * Test script using URLSearchParams for authentication
 */

const fetch = require('node-fetch');

// Test credentials - using the exact same credentials from ensure_test_user.py
const TEST_USER = {
  username: 'testuser',
  password: 'testpassword',
  email: 'test@example.com'
};

async function testLogin() {
  console.log(`Testing login with: ${TEST_USER.username}`);
  
  try {
    // Use URLSearchParams for form data
    const params = new URLSearchParams();
    params.append('username', TEST_USER.username);
    params.append('password', TEST_USER.password);
    
    // Make the login request
    console.log('Making request to: http://localhost:8000/api/login');
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params
    });
    
    console.log(`Response status: ${response.status} ${response.statusText}`);
    
    // Parse the response
    let data;
    try {
      data = await response.json();
      console.log('Response data:', data);
    } catch (error) {
      console.log('Error parsing response JSON:', error.message);
      console.log('Raw response:', await response.text());
      return null;
    }
    
    return response.ok ? data : null;
  } catch (error) {
    console.error('Login request error:', error.message);
    return null;
  }
}

// Start the test
(async () => {
  const result = await testLogin();
  
  if (result && result.session_id) {
    console.log('\n✅ SUCCESS: Authentication works!');
    console.log(`Session ID: ${result.session_id}`);
  } else {
    console.log('\n❌ FAILURE: Authentication failed.');
  }
})();
