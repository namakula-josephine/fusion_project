/**
 * Test script for verifying the API client functionality
 */

const axios = require('axios');
const FormData = require('form-data');

// Base URL for the API
const API_BASE_URL = 'http://localhost:8000';

// Test credentials
const TEST_USER = {
  username: 'testuser',
  password: 'testpassword',
  email: 'test@example.com'
};

/**
 * Login function that mimics the API client's login method
 */
async function login(credentials) {
  try {
    console.log('Logging in with:', credentials.username);
    
    // Create form data
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);

    // Try with axios directly
    console.log(`Making POST request to ${API_BASE_URL}/api/login`);
    const response = await axios.post(`${API_BASE_URL}/api/login`, formData, {
      headers: formData.getHeaders()
    });
    
    console.log('Login response status:', response.status);
    console.log('Login response data:', response.data);
    
    return response.data;
  } catch (error) {
    console.error('Login error:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return null;
  }
}

/**
 * Alternative login function using URLSearchParams
 */
async function loginWithURLSearchParams(credentials) {
  try {
    console.log('Logging in with URLSearchParams:', credentials.username);
    
    // Create URLSearchParams
    const params = new URLSearchParams();
    params.append('username', credentials.username);
    params.append('password', credentials.password);

    // Make the request
    console.log(`Making POST request to ${API_BASE_URL}/api/login`);
    const response = await axios.post(`${API_BASE_URL}/api/login`, params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    
    console.log('Login response status:', response.status);
    console.log('Login response data:', response.data);
    
    return response.data;
  } catch (error) {
    console.error('Login error:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return null;
  }
}

// Run the test
(async () => {
  // Try with FormData first
  console.log('\n=== Testing login with FormData ===');
  const formDataResult = await login(TEST_USER);
  
  if (formDataResult && formDataResult.session_id) {
    console.log('✅ FormData login successful!');
  } else {
    console.log('❌ FormData login failed.');
  }
  
  // Then try with URLSearchParams
  console.log('\n=== Testing login with URLSearchParams ===');
  const urlParamsResult = await loginWithURLSearchParams(TEST_USER);
  
  if (urlParamsResult && urlParamsResult.session_id) {
    console.log('✅ URLSearchParams login successful!');
  } else {
    console.log('❌ URLSearchParams login failed.');
  }
})();
