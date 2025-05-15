import "../../index.css";
import { ThemeProvider } from "../theme/ThemeProvider";

export const metadata = {
  title: "PotatoGuard - Potato Disease Detection",
  description: "Detect and treat late blight and early blight in potatoes",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}