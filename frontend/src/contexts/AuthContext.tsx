import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { api, User } from '../lib/api';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('ai-presentations-auth-token');
      if (token) {
        try {
          api.setToken(token);
          const me = await api.getMe();
          setUser(me);
        } catch (err) {
          // Token invalid, clear it
          api.setToken(null);
          localStorage.removeItem('ai-presentations-auth-token');
        }
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (username: string, password: string) => {
    try {
      setError(null);
      await api.login(username, password);
      const me = await api.getMe();
      setUser(me);
      localStorage.setItem('ai-presentations-username', username);
    } catch (err: any) {
      setError(err.message || 'Ошибка входа');
      throw err;
    }
  };

  const register = async (username: string, password: string) => {
    try {
      setError(null);
      await api.register(username, password);
      // After registration, login automatically
      await login(username, password);
    } catch (err: any) {
      setError(err.message || 'Ошибка регистрации');
      throw err;
    }
  };

  const logout = () => {
    api.setToken(null);
    setUser(null);
    localStorage.removeItem('ai-presentations-auth-token');
    localStorage.removeItem('ai-presentations-username');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
