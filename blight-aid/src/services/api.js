import authService from './authService';

/**
 * API Service for connecting to the Potato Blight RAG backend
 */

const apiService = {
  baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',

  // Helper to get headers with auth token
  getHeaders() {
    const token = authService.getToken();
    return {
      'Authorization': token ? `Bearer ${token}` : '',
    };
  },

  /**
   * Send a text query to the RAG system
   * @param {string} question - The user's question
   * @param {string} sessionId - The current session ID
   * @returns {Promise} - Response from the backend
   */
  async sendTextQuery(question, sessionId) {
    try {
      console.log(`Sending text query to RAG backend: ${question}`);
      
      const formData = new FormData();
      formData.append('question', question);
      formData.append('session_id', sessionId);

      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`RAG API Error: ${errorText}`);
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('RAG response received:', data);
      return data;
    } catch (error) {
      console.error('RAG API Error:', error);
      throw error;
    }
  },

  /**
   * Send an image for analysis to the vision model and RAG system
   * @param {File} imageFile - The image file to analyze
   * @param {string} sessionId - The current session ID
   * @returns {Promise} - Response from the backend
   */
  async analyzeImage(imageFile, sessionId) {
    try {
      console.log(`Sending image for analysis: ${imageFile.name}`);
      
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('session_id', sessionId);

      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Image Analysis Error: ${errorText}`);
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Image analysis received:', data);
      return data;
    } catch (error) {
      console.error('Image Analysis Error:', error);
      throw error;
    }
  },

  /**
   * Get chat history for a session
   * @param {string} sessionId - The session ID
   * @returns {Promise} - Chat history from the backend
   */
  async getChatHistory(sessionId) {
    try {
      console.log(`Fetching chat history for session: ${sessionId}`);
      
      const response = await fetch(`${this.baseUrl}/chat-history?session_id=${sessionId}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Chat History Error: ${errorText}`);
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Chat history received:', data);
      return data;
    } catch (error) {
      console.error('Chat History Error:', error);
      throw error;
    }
  }
};

export default apiService;