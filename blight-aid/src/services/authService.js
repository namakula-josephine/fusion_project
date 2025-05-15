const authService = {
  baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  tokenKey: 'potato_assistant_token',

  // Login method
  async login(email, password) {
    try {
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      this.setToken(data.token);
      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },

  // Signup method
  async signup(userData) {
    try {
      const response = await fetch(`${this.baseUrl}/auth/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        throw new Error('Signup failed');
      }

      const data = await response.json();
      this.setToken(data.token);
      return data;
    } catch (error) {
      console.error('Signup error:', error);
      throw error;
    }
  },

  // Password reset request
  async forgotPassword(email) {
    try {
      const response = await fetch(`${this.baseUrl}/auth/forgot-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      if (!response.ok) {
        throw new Error('Password reset request failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Forgot password error:', error);
      throw error;
    }
  },

  // Token management
  setToken(token) {
    localStorage.setItem(this.tokenKey, token);
  },

  getToken() {
    return localStorage.getItem(this.tokenKey);
  },

  isAuthenticated() {
    return !!this.getToken();
  },

  // Logout
  logout() {
    localStorage.removeItem(this.tokenKey);
    // Clear other storage items
    localStorage.removeItem('potato_assistant_session');
    localStorage.removeItem('potato_assistant_recent_chats');
  }
};

export default authService;