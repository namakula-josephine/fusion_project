import { Navigate } from 'react-router-dom';

export default function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token');

  // Ensure token is not just present but also valid
  const isValidToken = token && token.trim() !== '';

  if (!isValidToken) {
    return <Navigate to="/login" replace />;
  }

  return children || <div>No content provided for protected route</div>;
}