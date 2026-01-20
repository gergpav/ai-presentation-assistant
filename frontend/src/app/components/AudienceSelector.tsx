import { useState } from 'react';
import { Button } from './ui/button';
import { Users, TrendingUp, Award } from 'lucide-react';
import type { AudienceType } from '../App';

interface AudienceSelectorProps {
  onSelect: (audience: AudienceType) => void;
  uploadedFilesCount: number;
}

interface AudienceOption {
  type: AudienceType;
  title: string;
  description: string;
  icon: React.ElementType;
  features: string[];
  color: string;
}

const audiences: AudienceOption[] = [
  {
    type: 'executive',
    title: 'Топ-менеджмент',
    description: 'Краткие выводы и стратегические решения',
    icon: Users,
    features: [
      'Фокус на ключевых показателях',
      'Стратегические выводы',
      'Минимум технических деталей',
      'Акцент на бизнес-результаты'
    ],
    color: 'blue'
  },
  {
    type: 'expert',
    title: 'Эксперты',
    description: 'Подробный технический и аналитический контент',
    icon: Award,
    features: [
      'Детальный анализ данных',
      'Технические детали',
      'Методология расчетов',
      'Глубокая экспертиза'
    ],
    color: 'purple'
  },
  {
    type: 'investor',
    title: 'Инвесторы',
    description: 'Финансовые показатели и возврат инвестиций',
    icon: TrendingUp,
    features: [
      'Финансовые метрики',
      'Анализ рисков',
      'ROI и окупаемость',
      'Инвестиционная привлекательность'
    ],
    color: 'green'
  }
];

export function AudienceSelector({ onSelect, uploadedFilesCount }: AudienceSelectorProps) {
  const [selectedAudience, setSelectedAudience] = useState<AudienceType | null>(null);

  const handleSelect = (audienceType: AudienceType) => {
    setSelectedAudience(audienceType);
  };

  const handleContinue = () => {
    if (selectedAudience) {
      onSelect(selectedAudience);
    }
  };

  const getColorClasses = (color: string, isSelected: boolean) => {
    const colors = {
      blue: {
        border: isSelected ? 'border-blue-500' : 'border-gray-200',
        bg: isSelected ? 'bg-blue-50' : 'bg-white',
        icon: 'bg-blue-100 text-blue-600',
        checkmark: 'bg-blue-600'
      },
      purple: {
        border: isSelected ? 'border-purple-500' : 'border-gray-200',
        bg: isSelected ? 'bg-purple-50' : 'bg-white',
        icon: 'bg-purple-100 text-purple-600',
        checkmark: 'bg-purple-600'
      },
      green: {
        border: isSelected ? 'border-green-500' : 'border-gray-200',
        bg: isSelected ? 'bg-green-50' : 'bg-white',
        icon: 'bg-green-100 text-green-600',
        checkmark: 'bg-green-600'
      }
    };
    return colors[color as keyof typeof colors];
  };

  return (
    <div className="p-8">
      <div className="max-w-5xl mx-auto">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm mb-4">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            Загружено файлов: {uploadedFilesCount}
          </div>
          <h2 className="text-2xl mb-2 text-gray-900">Выберите целевую аудиторию</h2>
          <p className="text-gray-600">
            AI адаптирует глубину и стиль подачи материала под выбранную аудиторию
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {audiences.map(audience => {
            const isSelected = selectedAudience === audience.type;
            const colors = getColorClasses(audience.color, isSelected);
            const Icon = audience.icon;

            return (
              <div
                key={audience.type}
                onClick={() => handleSelect(audience.type)}
                className={`
                  relative border-2 rounded-lg p-6 cursor-pointer transition-all
                  ${colors.border} ${colors.bg}
                  hover:shadow-lg
                `}
              >
                {isSelected && (
                  <div className={`absolute top-4 right-4 w-6 h-6 ${colors.checkmark} rounded-full flex items-center justify-center`}>
                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}

                <div className={`w-12 h-12 ${colors.icon} rounded-lg flex items-center justify-center mb-4`}>
                  <Icon className="w-6 h-6" />
                </div>

                <h3 className="text-lg mb-2 text-gray-900">{audience.title}</h3>
                <p className="text-sm text-gray-600 mb-4">{audience.description}</p>

                <ul className="space-y-2">
                  {audience.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                      <svg className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>

        <div className="flex justify-center">
          <Button
            onClick={handleContinue}
            disabled={!selectedAudience}
            size="lg"
            className="px-8"
          >
            Начать создание презентации
          </Button>
        </div>
      </div>
    </div>
  );
}
