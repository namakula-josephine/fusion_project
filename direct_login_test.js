/**
 * Direct test script for FastAPI authentication
 */

const fetch = require('node-fetch');
const FormData = require('form-data');

// Match the credentials exactly with what's created by ensure_test_user.py
const TEST_USER = {
  username: 'testuser',
  password: 'testpassword',
  email: 'test@example.com'
};

async function directLogin() {
  try {
    console.log(`Testing direct login with ${TEST_USER.username}`);
    
    // Create form data for the request
    const formData = new FormData();
    formData.append('username', TEST_USER.username);
    formData.append('password', TEST_USER.password);
    
    // Make the request
    console.log('Sending request to http://localhost:8000/api/login');
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      body: formData
    });
    
    console.log(`Response status: ${response.status} ${response.statusText}`);
    
    // Parse response
    const data = await response.json().catch(() => {
      console.log('Failed to parse JSON response');
      return null;
    });
    
    console.log('Response data:', data);
    
    // Show result
    if (response.ok && data && data.session_id) {
      console.log('\n✅ SUCCESS: Authentication works!');
      console.log(`Session ID: ${data.session_id}`);
    } else {
      console.log('\n❌ FAILURE: Authentication failed.');
    }
  } catch (error) {
    console.error('Error during login test:', error);
  }
}

// Run the test
directLogin();
