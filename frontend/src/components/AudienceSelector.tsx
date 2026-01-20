import { Users } from './Icons';
import { SimpleLabel } from './SimpleLabel';

interface AudienceSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function AudienceSelector({ value, onChange }: AudienceSelectorProps) {
  return (
    <div className="space-y-2">
      <SimpleLabel className="flex items-center gap-2">
        <Users className="w-4 h-4" />
        Целевая аудитория
      </SimpleLabel>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="flex h-9 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-[#0057B8] focus:ring-2 focus:ring-[#0057B8]/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="investors">Инвесторы</option>
        <option value="partners">Партнеры</option>
        <option value="clients">Клиенты</option>
      </select>
    </div>
  );
}
