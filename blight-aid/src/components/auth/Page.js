import { Button } from '../ui/button';
import { Link } from 'react-router-dom';

export default function Home() {
  const backgroundImageUrl = process.env.PUBLIC_URL + '/placeholder.svg';

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{
        backgroundImage: `url(${backgroundImageUrl}?height=1080&width=1920)`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        backgroundBlendMode: "overlay",
      }}
    >
      <header className="px-4 lg:px-6 h-16 flex items-center">
        <Link className="flex items-center justify-center" to="/">
          <span className="font-bold text-lg text-white">PotatoGuard</span>
        </Link>
        <nav className="ml-auto flex gap-4 sm:gap-6">
          <Link to="/login" className="text-sm font-medium text-white hover:text-green-200">
            Login
          </Link>
          <Link
            to="/signup"
            className="text-sm font-medium bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-md"
          >
            Sign Up
          </Link>
        </nav>
      </header>

      <main className="flex-1 flex items-center">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none text-white">
                Detect & Treat Potato Diseases
              </h1>
              <p className="max-w-[700px] text-white/80 md:text-xl mx-auto">
                Early detection of late blight and early blight diseases in potatoes. Get AI-powered treatment plans
                using advanced RAG technology.
              </p>
            </div>
            <div className="flex flex-col gap-2 min-[400px]:flex-row">
              <Link to="/signup">
                <Button size="lg" className="bg-green-600 hover:bg-green-700">
                  Get Started
                </Button>
              </Link>
              <Link to="/login">
                <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10">
                  Login
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </main>

      <footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t border-white/10">
        <p className="text-xs text-white/60">© {new Date().getFullYear()} PotatoGuard. All rights reserved.</p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link className="text-xs text-white/60 hover:text-white" to="/terms">
            Terms of Service
          </Link>
          <Link className="text-xs text-white/60 hover:text-white" to="/privacy">
            Privacy
          </Link>
        </nav>
      </footer>
    </div>
  )
}

