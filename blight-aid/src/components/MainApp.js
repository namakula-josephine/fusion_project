import React, { useState } from 'react';
import { Leaf, Upload, X } from "lucide-react";
import { Button } from './ui/button';
import { useNavigate } from 'react-router-dom';

export default function MainApp() {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-green-600 text-white p-4">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <Leaf className="h-6 w-6 mr-2" />
            <span className="font-bold text-lg">PotatoGuard</span>
          </div>
          <Button 
            variant="ghost" 
            className="text-white hover:text-green-200"
            onClick={handleLogout}
          >
            Logout
          </Button>
        </div>
      </nav>

      <main className="container mx-auto p-6 max-w-4xl">
        <h1 className="text-3xl font-bold mb-8">Potato Disease Detection</h1>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8">
              {preview ? (
                <div className="relative">
                  <img 
                    src={preview} 
                    alt="Preview" 
                    className="max-h-[400px] mx-auto rounded"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setPreview(null);
                      setSelectedImage(null);
                      setResult(null);
                    }}
                    className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <Upload className="mx-auto h-12 w-12 text-gray-400" />
                  <div className="mt-4">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => document.getElementById('file-upload').click()}
                    >
                      Upload Image
                    </Button>
                    <input
                      id="file-upload"
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageChange}
                    />
                  </div>
                  <p className="mt-2 text-sm text-gray-500">
                    Upload a clear image of the potato plant leaf
                  </p>
                </div>
              )}
            </div>

            {preview && (
              <Button 
                type="submit" 
                className="w-full bg-green-600 hover:bg-green-700"
                disabled={isLoading}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Image'}
              </Button>
            )}
          </form>

          {result && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h2 className="text-xl font-semibold mb-4">Analysis Result</h2>
              <pre className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}