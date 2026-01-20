import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip';
import { Sparkles, Download, ChevronLeft, Edit2, Check, Users, Award, TrendingUp, FileText } from 'lucide-react';
import type { Presentation } from '../../lib/types';

interface PresentationHeaderProps {
  presentation: Presentation;
  onUpdateTitle: (title: string) => void;
  onExport: () => void;
  onReset: () => void;
  canExport: boolean;
}

export function PresentationHeader({
  presentation,
  onUpdateTitle,
  onExport,
  onReset,
  canExport
}: PresentationHeaderProps) {
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState(presentation.title);

  const handleSaveTitle = () => {
    if (editTitle.trim()) {
      onUpdateTitle(editTitle.trim());
      setIsEditingTitle(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveTitle();
    } else if (e.key === 'Escape') {
      setEditTitle(presentation.title);
      setIsEditingTitle(false);
    }
  };

  const getAudienceInfo = () => {
    const audienceMap = {
      executive: { label: 'Топ-менеджмент', icon: Users, color: 'blue' },
      expert: { label: 'Эксперты', icon: Award, color: 'purple' },
      investor: { label: 'Инвесторы', icon: TrendingUp, color: 'green' }
    };
    return audienceMap[presentation.audience];
  };

  const audienceInfo = getAudienceInfo();
  const AudienceIcon = audienceInfo.icon;

  const pendingSlides = presentation.slides.filter(s => s.status === 'pending').length;
  const generatingSlides = presentation.slides.filter(s => s.isGenerating).length;

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="gap-2"
          >
            <ChevronLeft className="w-4 h-4" />
            Назад
          </Button>

          <div className="h-6 w-px bg-gray-300" />

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>

            {isEditingTitle ? (
              <div className="flex items-center gap-2">
                <Input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onKeyDown={handleKeyPress}
                  className="w-96"
                  autoFocus
                  onBlur={handleSaveTitle}
                />
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={handleSaveTitle}
                >
                  <Check className="w-4 h-4" />
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <h1 className="text-xl text-gray-900">{presentation.title}</h1>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setIsEditingTitle(true)}
                  className="opacity-0 hover:opacity-100 transition-opacity"
                >
                  <Edit2 className="w-3 h-3" />
                </Button>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className={`
            flex items-center gap-2 px-3 py-1.5 rounded-full text-sm
            ${audienceInfo.color === 'blue' ? 'bg-blue-100 text-blue-700' : ''}
            ${audienceInfo.color === 'purple' ? 'bg-purple-100 text-purple-700' : ''}
            ${audienceInfo.color === 'green' ? 'bg-green-100 text-green-700' : ''}
          `}>
            <AudienceIcon className="w-4 h-4" />
            {audienceInfo.label}
          </div>

          {presentation.template && (
            <>
              <div className="h-6 w-px bg-gray-300" />
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-50 text-amber-700 rounded-full text-sm">
                      <FileText className="w-4 h-4" />
                      Шаблон
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{presentation.template.name}</p>
                    <p className="text-xs text-gray-400">
                      {(presentation.template.size / 1024).toFixed(1)} KB
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          )}

          <div className="h-6 w-px bg-gray-300" />

          <div className="text-sm text-gray-600">
            <span className="font-medium">{presentation.slides.length}</span> слайдов
          </div>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span>
                  <Button className="gap-2" onClick={onExport} disabled={!canExport}>
                    <Download className="w-4 h-4" />
                    Экспортировать
                  </Button>
                </span>
              </TooltipTrigger>
              {!canExport && (
                <TooltipContent>
                  <p className="max-w-xs">
                    {presentation.slides.length === 0
                      ? 'Добавьте хотя бы один слайд для экспорта'
                      : generatingSlides > 0
                      ? `Дождитесь завершения генерации (${generatingSlides} слайд${generatingSlides > 1 ? 'ов' : ''})`
                      : `Сгенерируйте все слайды перед экспортом (осталось: ${pendingSlides})`
                    }
                  </p>
                </TooltipContent>
              )}
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </header>
  );
}