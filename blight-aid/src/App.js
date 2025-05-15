import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './components/auth/login';
import SignUp from './components/auth/SignUp';
import ForgotPassword from './components/auth/ForgotPassword';
import AuthLayout from './components/auth/AuthLayout';
import ErrorBoundary from './components/ErrorBoundary';
import ProtectedRoute from './components/ProtectedRoute';
import { v4 as uuidv4 } from 'uuid';
import './App.css';
import './index.css';

// API Service
const apiService = {
  baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',

  // Method to handle text queries
  async sendTextQuery(question, sessionId) {
    try {
      const formData = new FormData();
      formData.append('question', question);
      formData.append('session_id', sessionId);

      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },

  // Method to handle image uploads and analysis
  async analyzeImage(imageFile, sessionId) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('session_id', sessionId);

      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
};

// Session Service
const sessionService = {
  storageKey: 'potato_assistant_session',
  chatsKey: 'potato_assistant_recent_chats',

  // Get the current session ID or create a new one
  getSessionId() {
    const storedSession = localStorage.getItem(this.storageKey);
    
    if (storedSession) {
      try {
        const sessionData = JSON.parse(storedSession);
        return sessionData.id;
      } catch (e) {
        console.error('Error parsing session data:', e);
      }
    }
    
    // Create a new session if none exists
    return this.createNewSession();
  },

  // Create a new session
  createNewSession() {
    const sessionId = uuidv4();
    
    const sessionData = {
      id: sessionId,
      createdAt: new Date().toISOString()
    };
    
    localStorage.setItem(this.storageKey, JSON.stringify(sessionData));
    return sessionId;
  },

  // Store recent chats in local storage
  saveRecentChats(chats) {
    localStorage.setItem(this.chatsKey, JSON.stringify(chats));
  },

  // Get recent chats from local storage
  getRecentChats() {
    const storedChats = localStorage.getItem(this.chatsKey);
    
    if (storedChats) {
      try {
        return JSON.parse(storedChats);
      } catch (e) {
        console.error('Error parsing recent chats:', e);
        return [];
      }
    }
    
    return [];
  }
};

function App() {
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Use the session service to get or create a session ID
  const [sessionId] = useState(sessionService.getSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('text');
  // Load recent chats from session service
  const [recentChats, setRecentChats] = useState(
    sessionService.getRecentChats() || [
      { id: '1', title: 'Early Blight Analysis', description: 'Analysis from yesterday' },
      { id: '2', title: 'Late Blight Treatment', description: 'Treatment options discussion' },
      { id: '3', title: 'Healthy Plant Confirmation', description: 'Verification of plant health' }
    ]
  );
  const [currentChatTitle, setCurrentChatTitle] = useState('');
  const [chatStarted, setChatStarted] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  // Save recent chats to local storage when they change
  useEffect(() => {
    sessionService.saveRecentChats(recentChats);
  }, [recentChats]);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update chat title based on first message
  useEffect(() => {
    if (messages.length === 1 && messages[0].role === 'user') {
      // Create a title from the first message
      const firstMessage = messages[0].content;
      const title = firstMessage.length > 25 
        ? firstMessage.substring(0, 25) + '...' 
        : firstMessage;
      setCurrentChatTitle(title);
      setChatStarted(true);
    }
  }, [messages]);

  // Handle file upload
  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null); // Clear any previous errors
    }
  };

  // Handle file upload button click
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // Start a new chat
  const handleNewChat = () => {
    // Save current chat to recent chats if it has messages
    if (messages.length > 0 && currentChatTitle) {
      const newChat = {
        id: uuidv4(),
        title: currentChatTitle,
        description: `${messages.length} messages`,
        timestamp: new Date().toISOString()
      };
      
      setRecentChats(prev => [newChat, ...prev.slice(0, 4)]);
    }
    
    // Reset current chat
    setMessages([]);
    setCurrentChatTitle('');
    setChatStarted(false);
    setActiveTab('text');
    setError(null);
  };

  // Handle image analysis
  const handleAnalyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    
    // If this is the first message, start a new chat
    if (messages.length === 0) {
      setChatStarted(true);
      setCurrentChatTitle(`Analysis of ${file.name}`);
    }
    
    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: `I've uploaded an image for analysis: ${file.name}`,
      timestamp: new Date().toISOString()
    }]);

    try {
      // Use the API service
      const result = await apiService.analyzeImage(file, sessionId);
      
      // Add assistant message with results
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Here's my analysis of your potato plant:",
        result: result,
        timestamp: new Date().toISOString()
      }]);
      
      // Switch to text tab after analysis to allow continued conversation
      setActiveTab('text');
    } catch (error) {
      console.error('Image analysis error:', error);
      
      // Add error message
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: error.message || "Error processing the image",
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
      setFile(null);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage = input;
    setInput('');
    setError(null);
    
    // If this is the first message, start a new chat
    if (messages.length === 0) {
      setChatStarted(true);
      setCurrentChatTitle(userMessage.length > 25 
        ? userMessage.substring(0, 25) + '...' 
        : userMessage);
    }
    
    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);
    
    setLoading(true);
    
    try {
      // Use the API service
      const data = await apiService.sendTextQuery(userMessage, sessionId);
      
      // Add assistant message
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer || "Sorry, I couldn't process your question.",
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Text query error:', error);
      
      // Add error message
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: error.message || "Error connecting to the API",
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Load a chat history
  const handleLoadChat = async (chatId) => {
    try {
      setLoading(true);
      setError(null);
      
      // Find the chat in recent chats
      const chat = recentChats.find(c => c.id === chatId);
      
      if (chat) {
        // In a real app, you would fetch the actual chat history from your backend
        // For now, we'll just show a placeholder
        setMessages([
          {
            role: 'user',
            content: 'Loading previous chat...',
            timestamp: new Date().toISOString()
          },
          {
            role: 'assistant',
            content: 'This is where your previous chat history would appear. In a production app, you would load the actual messages from your backend.',
            timestamp: new Date().toISOString()
          }
        ]);
        
        setCurrentChatTitle(chat.title);
        setChatStarted(true);
      }
    } catch (error) {
      console.error('Load chat error:', error);
      setError(error.message || "Error loading chat history");
    } finally {
      setLoading(false);
    }
  };

  const handleAuthentication = (token) => {
    localStorage.setItem('token', token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setMessages([]);
    setCurrentChatTitle('');
    setChatStarted(false);
  };

  // Render message
  const renderMessage = (message, index) => {
    return (
      <div 
        key={index} 
        className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
      >
        <div className="message-avatar">
          {message.role === 'user' ? '👤' : '🤖'}
        </div>
        <div className="message-content">
          <p>{message.content}</p>
          
          {message.result && (
            <div className="message-result">
              <div className="result-item">
                <strong>Prediction:</strong> {message.result.predicted_class}
              </div>
              <div className="result-item">
                <strong>Confidence:</strong> {message.result.confidence}
              </div>
              <div className="result-item">
                <strong>Explanation:</strong> {message.result.explanation}
              </div>
              <div className="result-item">
                <strong>Treatment Plans:</strong> {message.result.treatment_plans}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render error message
  const renderError = () => {
    if (!error) return null;
    
    return (
      <div className="error-message">
        <div className="error-icon">⚠️</div>
        <div className="error-content">
          <h3>Something went wrong</h3>
          <p>{error}</p>
          <button 
            className="retry-button" 
            onClick={() => setError(null)}
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  };

  // Main app component that contains your existing UI
  const MainAppContent = () => {
    return (
      <div className="app">
        <div className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo">🥔</div>
            <h1>Potato Assistant</h1>
          </div>
          
          <div className="sidebar-section">
            <h2>Folders</h2>
            <div className="folder-item">
              <div className="folder-icon">📁</div>
              <span>Saved Analyses</span>
            </div>
            <div className="folder-item">
              <div className="folder-icon">📁</div>
              <span>Treatment Plans</span>
            </div>
            <div className="folder-item">
              <div className="folder-icon">📁</div>
              <span>Disease Info</span>
            </div>
            <div className="folder-item">
              <div className="folder-icon">📁</div>
              <span>My History</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <h2>Recent Chats</h2>
            {recentChats.map(chat => (
              <div 
                key={chat.id} 
                className="chat-item"
                onClick={() => handleLoadChat(chat.id)}
              >
                <div className="chat-icon">💬</div>
                <div className="chat-item-content">
                  <div className="chat-item-title">{chat.title}</div>
                  <div className="chat-item-description">{chat.description}</div>
                </div>
              </div>
            ))}
            
            {chatStarted && (
              <div className="chat-item active-chat">
                <div className="chat-icon">🟢</div>
                <div className="chat-item-content">
                  <div className="chat-item-title">{currentChatTitle}</div>
                  <div className="chat-item-description">Current conversation</div>
                </div>
              </div>
            )}
          </div>
          
          <button className="new-chat-button" onClick={handleNewChat}>
            <span>+</span>
            <span>New Chat</span>
          </button>
          <button className="logout-button" onClick={handleLogout}>
            <span>🚪</span>
            <span>Logout</span>
          </button>
        </div>
        
        <div className="main-content">
          {chatStarted && (
            <div className="chat-header">
              <h2>{currentChatTitle}</h2>
            </div>
          )}
          
          <div className="chat-container">
            {renderError()}
            
            {messages.length === 0 ? (
              <div className="welcome-container">
                <div className="welcome-icon">🥔</div>
                <h1>How can I help you today?</h1>
                <p>
                  This tool can analyze potato plant images for diseases and answer 
                  questions about potato plant care. Upload an image or ask a question below.
                </p>
                
                <div className="feature-cards">
                  <div className="feature-card">
                    <div className="feature-icon">📊</div>
                    <h3>Disease Classification</h3>
                    <p>Upload images to detect Early Blight, Late Blight, or confirm healthy plants</p>
                  </div>
                  <div className="feature-card">
                    <div className="feature-icon">💊</div>
                    <h3>Treatment Recommendations</h3>
                    <p>Get personalized treatment plans based on disease diagnosis</p>
                  </div>
                  <div className="feature-card">
                    <div className="feature-icon">🔍</div>
                    <h3>Expert Knowledge</h3>
                    <p>Access information about potato diseases, prevention, and care</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="messages">
                {messages.map(renderMessage)}
                <div ref={messagesEndRef} />
                
                {loading && (
                  <div className="message assistant-message">
                    <div className="message-avatar">🤖</div>
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* File Upload Section */}
          <div className="upload-section">
            {activeTab === 'image' && (
              <div className="file-upload">
                <input 
                  type="file" 
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                  accept="image/*" 
                  style={{ display: 'none' }}
                />
                
                <div className="file-upload-content">
                  {file ? (
                    <>
                      <div className="file-name">{file.name}</div>
                      <button 
                        className="analyze-button"
                        onClick={handleAnalyzeImage}
                        disabled={loading}
                      >
                        {loading ? 'Analyzing...' : 'Analyze Image'}
                      </button>
                    </>
                  ) : (
                    <button 
                      className="upload-button"
                      onClick={handleUploadClick}
                    >
                      <span>📤</span>
                      <span>Upload Image</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* Input Section */}
          <div className="input-section">
            <div className="tabs">
              <button 
                className={`tab-button ${activeTab === 'text' ? 'active' : ''}`}
                onClick={() => setActiveTab('text')}
              >
                Text
              </button>
              <button 
                className={`tab-button ${activeTab === 'image' ? 'active' : ''}`}
                onClick={() => setActiveTab('image')}
              >
                <span>📷</span>
                <span>Image</span>
              </button>
              <button 
                className={`tab-button ${activeTab === 'analytics' ? 'active' : ''}`}
                onClick={() => setActiveTab('analytics')}
              >
                <span>📊</span>
                <span>Analytics</span>
              </button>
            </div>
            
            <form onSubmit={handleSubmit} className="input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your question here..."
                disabled={loading || activeTab !== 'text'}
              />
              <button 
                type="submit" 
                disabled={loading || !input.trim() || activeTab !== 'text'}
                className="send-button"
              >
                ➤
              </button>
            </form>
            
            <div className="disclaimer">
              Powered by FastAPI, TensorFlow, and OpenAI GPT-4
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Router>
      <Routes>
        {/* Auth Routes */}
        <Route element={<AuthLayout />}>
          <Route 
            path="/login" 
            element={
              !isAuthenticated ? (
                <Login onLogin={handleAuthentication} />
              ) : (
                <Navigate to="/" replace />
              )
            } 
          />
          <Route 
            path="/signup" 
            element={
              !isAuthenticated ? (
                <SignUp onSignUp={handleAuthentication} />
              ) : (
                <Navigate to="/" replace />
              )
            } 
          />
          <Route 
            path="/forgot-password" 
            element={<ForgotPassword />} 
          />
        </Route>

        {/* Main App Route */}
        <Route 
          path="/" 
          element={
            isAuthenticated ? (
              <MainAppContent />
            ) : (
              <Navigate to="/login" replace />
            )
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;