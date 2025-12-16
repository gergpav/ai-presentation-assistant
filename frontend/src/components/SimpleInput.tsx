interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function SimpleInput({ className = '', ...props }: InputProps) {
  return (
    <input
      className={`flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-[#0057B8] focus:ring-2 focus:ring-[#0057B8]/20 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...props}
    />
  );
}

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export function SimpleTextarea({ className = '', ...props }: TextareaProps) {
  return (
    <textarea
      className={`flex w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-[#0057B8] focus:ring-2 focus:ring-[#0057B8]/20 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...props}
    />
  );
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {}

export function SimpleSelect({ className = '', ...props }: SelectProps) {
  return (
    <select
      className={`flex h-9 rounded-md border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-[#0057B8] focus:ring-2 focus:ring-[#0057B8]/20 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...props}
    />
  );
}
