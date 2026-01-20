import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Sparkles, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

export function LoginScreen() {
  const { login, register, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isRegistering, setIsRegistering] = useState(false);
  const [passwordError, setPasswordError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError('');

    if (!username.trim() || password.length < 6) {
      return;
    }

    // Проверка совпадения паролей при регистрации
    if (isRegistering) {
      if (password !== confirmPassword) {
        setPasswordError('Пароли не совпадают');
        return;
      }
      try {
        await register(username.trim(), password);
      } catch (err) {
        // Error handled by AuthContext
      }
    } else {
      try {
        await login(username.trim(), password);
      } catch (err) {
        // Error handled by AuthContext
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl mb-2 text-gray-900">AI Презентации</h1>
          <p className="text-gray-600">
            {isRegistering ? 'Создайте аккаунт для начала работы' : 'Войдите в ваш аккаунт'}
          </p>
        </div>

        {/* Login Form */}
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username */}
            <div className="space-y-2">
              <Label htmlFor="username">
                {isRegistering ? 'Придумайте логин' : 'Логин'}
              </Label>
              <Input
                id="username"
                type="text"
                placeholder={isRegistering ? 'Введите логин' : 'Ваш логин'}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                required
              />
            </div>

            {/* Password */}
            <div className="space-y-2">
              <Label htmlFor="password">
                {isRegistering ? 'Придумайте пароль' : 'Пароль'}
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder={isRegistering ? 'Минимум 6 символов' : 'Ваш пароль'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            {/* Confirm Password */}
            {isRegistering && (
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Подтвердите пароль</Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Повторите пароль"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>
                {passwordError && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                    {passwordError}
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full h-11"
              disabled={!username.trim() || password.length < 6}
            >
              {isRegistering ? 'Зарегистрироваться' : 'Войти'}
            </Button>
          </form>

          {/* Toggle Register/Login */}
          <div className="mt-6 text-center">
            <button
              onClick={() => {
                setIsRegistering(!isRegistering);
                setPasswordError('');
                setConfirmPassword('');
              }}
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              {isRegistering ? (
                <>
                  Уже есть аккаунт? <span className="text-blue-600 font-medium">Войти</span>
                </>
              ) : (
                <>
                  Нет аккаунта? <span className="text-blue-600 font-medium">Зарегистрироваться</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}