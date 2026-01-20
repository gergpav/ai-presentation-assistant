import { useState } from 'react';
import { Button } from './ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { Trash2, Users, Award, TrendingUp, FileText, Clock } from 'lucide-react';
import type { Presentation } from '../../lib/types';

interface ProjectCardProps {
  project: Presentation;
  onOpen: () => void;
  onDelete: () => void;
}

export function ProjectCard({ project, onOpen, onDelete }: ProjectCardProps) {
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const audienceConfig = {
    executive: { label: 'Топ-менеджмент', icon: Users, color: 'blue' },
    expert: { label: 'Эксперты', icon: Award, color: 'purple' },
    investor: { label: 'Инвесторы', icon: TrendingUp, color: 'green' }
  };

  const config = audienceConfig[project.audience];
  const AudienceIcon = config.icon;

  const completedSlides = project.slides.filter(s => s.status === 'completed').length;
  const totalSlides = project.slides.length;
  const progress = totalSlides > 0 ? (completedSlides / totalSlides) * 100 : 0;

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - new Date(date).getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Сегодня';
    if (days === 1) return 'Вчера';
    if (days < 7) return `${days} дн. назад`;
    
    return new Date(date).toLocaleDateString('ru-RU', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  };

  return (
    <>
      <div
        onClick={onOpen}
        className="group bg-white rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-lg transition-all cursor-pointer overflow-hidden"
      >
        {/* Header */}
        <div className="p-5 border-b border-gray-100">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1 min-w-0">
              <h3 className="font-medium text-gray-900 mb-2 truncate group-hover:text-blue-600 transition-colors">
                {project.title}
              </h3>
              <div className={`
                inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
                ${config.color === 'blue' ? 'bg-blue-100 text-blue-700' : ''}
                ${config.color === 'purple' ? 'bg-purple-100 text-purple-700' : ''}
                ${config.color === 'green' ? 'bg-green-100 text-green-700' : ''}
              `}>
                <AudienceIcon className="w-3.5 h-3.5" />
                {config.label}
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 hover:bg-red-100"
              onClick={(e) => {
                e.stopPropagation();
                setShowDeleteDialog(true);
              }}
              title="Удалить проект"
            >
              <Trash2 className="w-4 h-4 text-red-600" />
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="p-5 space-y-4">
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-1.5">
              <FileText className="w-4 h-4" />
              <span>{totalSlides} {totalSlides === 1 ? 'слайд' : 'слайдов'}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Clock className="w-4 h-4" />
              <span>{formatDate(project.updatedAt)}</span>
            </div>
          </div>

          {/* Progress */}
          {totalSlides > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Прогресс</span>
                <span className="font-medium text-gray-900">
                  {completedSlides}/{totalSlides}
                </span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Удалить проект?</AlertDialogTitle>
            <AlertDialogDescription>
              Вы уверены, что хотите удалить проект "{project.title}"? Это действие нельзя отменить.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Отмена</AlertDialogCancel>
            <AlertDialogAction
              onClick={onDelete}
              className="bg-red-600 hover:bg-red-700 focus:ring-red-600"
            >
              Удалить
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
