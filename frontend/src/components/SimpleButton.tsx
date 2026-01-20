interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm';
  children: React.ReactNode;
}

export function SimpleButton({ 
  variant = 'default', 
  size = 'default',
  className = '',
  children,
  ...props 
}: ButtonProps) {
  const baseStyles = 'inline-flex items-center justify-center rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-[#0057B8]/20 disabled:opacity-50 disabled:pointer-events-none';
  
  const variantStyles = {
    default: 'bg-[#0057B8] text-white hover:bg-[#0057B8]/90',
    outline: 'border border-gray-300 bg-white hover:bg-gray-50',
    ghost: 'hover:bg-gray-100',
  };
  
  const sizeStyles = {
    default: 'h-9 px-4 py-2',
    sm: 'h-8 px-3 text-sm',
  };
  
  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
