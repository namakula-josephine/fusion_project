import React from 'react';
import { Outlet } from 'react-router-dom';
import { ThemeProvider } from "../theme/ThemeProvider";

function AuthLayout() {
  return (
    <div className="auth-layout">
      <ThemeProvider>
        <main className="auth-main">
          <Outlet />
        </main>
      </ThemeProvider>
    </div>
  );
}

export default AuthLayout;