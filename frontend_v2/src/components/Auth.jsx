import { useState } from 'react'
import { isSupabaseConfigured, supabase } from '../supabaseClient'
import { Mail, Lock, Loader2 } from 'lucide-react'

export default function Auth() {
  const [loading, setLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState(
    isSupabaseConfigured
      ? ''
      : 'Authentication is unavailable because Supabase environment variables are missing.'
  )
  const [messageType, setMessageType] = useState('error')

  const showMessage = (text, type = 'error') => {
    setMessage(text)
    setMessageType(type)
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    if (!isSupabaseConfigured) return
    setLoading(true)
    setMessage('')
    setMessageType('error')
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) {
      showMessage(error.message, 'error')
    } else {
      showMessage('Login successful.', 'success')
    }
    setLoading(false)
  }

  const handleSignUp = async (e) => {
    e.preventDefault()
    if (!isSupabaseConfigured) return
    setLoading(true)
    setMessage('')
    setMessageType('error')
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: window.location.origin,
      },
    })

    if (error) {
      showMessage(error.message, 'error')
    } else if (data?.session) {
      showMessage('Signup successful. You are now logged in.', 'success')
    } else {
      showMessage('Signup successful. Check your email for the confirmation link.', 'success')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d1117] text-[#c9d1d9] p-4">
      <div className="w-full max-w-md bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-8 rounded-lg shadow-2xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
            Financial Pragmatic AI
          </h1>
          <p className="text-[#8b949e] mt-2 italic text-sm">Secure Insight Intelligence</p>
        </div>

        <form className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">Email Address</label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-[#8b949e]">
                <Mail size={18} />
              </span>
              <input
                type="email"
                placeholder="email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#0d1117] border border-[#30363d] focus:border-blue-500 p-2 pl-10 rounded transition-all outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Password</label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-[#8b949e]">
                <Lock size={18} />
              </span>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0d1117] border border-[#30363d] focus:border-blue-500 p-2 pl-10 rounded transition-all outline-none"
              />
            </div>
          </div>

          <div className="flex flex-col gap-3 pt-2">
            <button
              onClick={handleLogin}
              type="button"
              disabled={loading || !isSupabaseConfigured}
              className="w-full bg-gradient-to-r from-blue-500 to-blue-700 hover:scale-[1.02] active:scale-95 transition-all py-2.5 rounded font-semibold text-white flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : 'Login'}
            </button>
            <button
              onClick={handleSignUp}
              type="button"
              disabled={loading || !isSupabaseConfigured}
              className="w-full bg-[#30363d] hover:bg-[#3a4149] py-2.5 rounded font-semibold transition-all text-sm"
            >
              Sign Up
            </button>
          </div>

          {message && (
            <div className={`text-sm mt-4 p-3 rounded border ${messageType === 'success' ? 'border-blue-500/30 bg-blue-500/10 text-blue-400' : 'border-red-500/30 bg-red-500/10 text-red-400'}`}>
              {message}
            </div>
          )}
        </form>
      </div>
    </div>
  )
}
