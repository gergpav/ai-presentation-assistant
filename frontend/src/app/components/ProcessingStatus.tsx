import { Progress } from './ui/progress';
import { Loader2, FileSearch, Database, Sparkles, FileText } from 'lucide-react';

interface ProcessingStatusProps {
  progress: number;
  currentStage: string;
}

export function ProcessingStatus({ progress, currentStage }: ProcessingStatusProps) {
  const stages = [
    {
      icon: FileSearch,
      title: 'Парсинг документов',
      description: 'Анализ структуры загруженных файлов',
      range: [0, 30]
    },
    {
      icon: Database,
      title: 'Извлечение данных',
      description: 'Извлечение таблиц, текстов и ключевых тезисов',
      range: [30, 60]
    },
    {
      icon: Sparkles,
      title: 'Генерация контента',
      description: 'AI создает слайды на основе данных',
      range: [60, 90]
    },
    {
      icon: FileText,
      title: 'Формирование презентации',
      description: 'Сборка финальной структуры',
      range: [90, 100]
    }
  ];

  const getCurrentStageIndex = () => {
    return stages.findIndex(stage => 
      progress >= stage.range[0] && progress < stage.range[1]
    );
  };

  const currentStageIndex = getCurrentStageIndex();

  return (
    <div className="p-12">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-blue-100 rounded-full mb-6 animate-pulse">
            <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
          </div>
          <h2 className="text-2xl mb-2 text-gray-900">Обработка документов</h2>
          <p className="text-gray-600">{currentStage}</p>
        </div>

        {/* Progress Bar */}
        <div className="mb-12">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-600">Прогресс</span>
            <span className="text-gray-900">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Stages */}
        <div className="space-y-4">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            const isCompleted = progress > stage.range[1];
            const isCurrent = index === currentStageIndex;
            const isPending = progress < stage.range[0];

            return (
              <div
                key={index}
                className={`
                  flex items-start gap-4 p-4 rounded-lg border transition-all
                  ${isCurrent ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200'}
                `}
              >
                <div className={`
                  w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0
                  ${isCompleted ? 'bg-green-100' : isCurrent ? 'bg-blue-100' : 'bg-gray-100'}
                `}>
                  {isCompleted ? (
                    <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <Icon className={`
                      w-5 h-5
                      ${isCurrent ? 'text-blue-600' : 'text-gray-400'}
                      ${isCurrent ? 'animate-pulse' : ''}
                    `} />
                  )}
                </div>

                <div className="flex-1">
                  <h3 className={`
                    mb-1
                    ${isCurrent ? 'text-blue-900' : isCompleted ? 'text-gray-900' : 'text-gray-500'}
                  `}>
                    {stage.title}
                  </h3>
                  <p className={`text-sm ${isCurrent || isCompleted ? 'text-gray-600' : 'text-gray-400'}`}>
                    {stage.description}
                  </p>
                  
                  {isCurrent && (
                    <div className="mt-2">
                      <Progress 
                        value={((progress - stage.range[0]) / (stage.range[1] - stage.range[0])) * 100} 
                        className="h-1"
                      />
                    </div>
                  )}
                </div>

                {isCompleted && (
                  <span className="text-xs text-green-600 flex-shrink-0">
                    Завершено
                  </span>
                )}
                {isCurrent && (
                  <span className="text-xs text-blue-600 flex-shrink-0">
                    Выполняется...
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-900">
            💡 <span className="font-medium">Совет:</span> AI анализирует структуру документов и 
            создает персонализированную презентацию на основе выбранной аудитории.
          </p>
        </div>
      </div>
    </div>
  );
}
