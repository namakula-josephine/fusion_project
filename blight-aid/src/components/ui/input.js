import React from 'react';

export function Input({ className, ...props }) {
  return (
    <input
      className={`border border-gray-300 rounded px-3 py-2 focus:ring-2 focus:ring-green-500 ${className}`}
      {...props}
    />
  );
}