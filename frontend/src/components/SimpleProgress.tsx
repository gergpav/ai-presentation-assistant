interface ProgressProps {
  value: number;
  className?: string;
}

export function SimpleProgress({ value, className = '' }: ProgressProps) {
  return (
    <div className={`w-full bg-gray-200 rounded-full h-2 overflow-hidden ${className}`}>
      <div
        className="bg-[#0057B8] h-full transition-all duration-300"
        style={{ width: `${value}%` }}
      />
    </div>
  );
}
