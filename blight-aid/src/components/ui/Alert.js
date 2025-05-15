import React from 'react';

export function Alert({ children, className, ...props }) {
  return (
    <div
      className={`p-4 border-l-4 bg-yellow-50 border-yellow-400 text-yellow-700 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function AlertDescription({ children, className, ...props }) {
  return (
    <p className={`text-sm ${className}`} {...props}>
      {children}
    </p>
  );
}