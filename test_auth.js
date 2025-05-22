// Check authentication system
const fetch = require('node-fetch');
const FormData = require('form-data');

async function registerUser(username, password, email) {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    formData.append('email', email);

    console.log(`Registering user: ${username}`);
    
    const response = await fetch('http://localhost:8000/api/register', {
      method: 'POST',
      body: formData
    });
    
    console.log('Registration status:', response.status);
    
    const data = await response.json().catch(() => {
      console.log('No JSON response body for registration');
      return null;
    });
    
    console.log('Registration data:', data);
    return response.ok;
  } catch (error) {
    console.error('Error registering user:', error);
    return false;
  }
}

async function loginUser(username, password) {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    console.log(`Logging in with user: ${username}`);
    
    const response = await fetch('http://localhost:8000/api/login', {
      method: 'POST',
      body: formData
    });
    
    console.log('Login status:', response.status);
    
    const data = await response.json().catch(() => {
      console.log('No JSON response body for login');
      return null;
    });
    
    console.log('Login data:', data);
    return response.ok ? data : null;
  } catch (error) {
    console.error('Error logging in:', error);
    return null;
  }
}

async function testAuthFlow() {
  console.log("=== Testing Authentication Flow ===");
  
  // Test user credentials
  const username = "testuser3";
  const password = "testpassword3";
  const email = "test3@example.com";
  
  // Step 1: Register a new user
  const registrationSuccess = await registerUser(username, password, email);
  
  if (!registrationSuccess) {
    console.log("Failed to register new user, will try logging in anyway");
  }
  
  // Step 2: Login with the new user
  const loginData = await loginUser(username, password);
  
  if (loginData) {
    console.log("Login successful!");
    console.log("Session ID:", loginData.session_id);
  } else {
    console.log("Login failed!");
  }
  
  console.log("=== Auth Flow Test Complete ===");
}

// Run the test
testAuthFlow();
