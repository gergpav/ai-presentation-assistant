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
import { GripVertical, Trash2, FileText, Sparkles, Copy } from 'lucide-react';
import type { Slide } from '../../lib/types';

interface SlideListProps {
  slides: Slide[];
  selectedSlideId: number | null;
  onSelectSlide: (slideId: number) => void;
  onDeleteSlide: (slideId: number) => void;
  onDuplicateSlide: (slideId: number) => void;
  onReorderSlides: (startIndex: number, endIndex: number) => void;
}

export function SlideList({
  slides,
  selectedSlideId,
  onSelectSlide,
  onDeleteSlide,
  onDuplicateSlide,
  onReorderSlides,
}: SlideListProps) {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [slideToDelete, setSlideToDelete] = useState<Slide | null>(null);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
  };

  const handleDrop = (index: number) => {
    if (draggedIndex === null) return;
    onReorderSlides(draggedIndex, index);
    setDraggedIndex(null);
  };

  if (slides.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <FileText className="w-6 h-6 text-gray-400" />
          </div>
          <p className="text-sm text-gray-600">Нет слайдов</p>
          <p className="text-xs text-gray-500 mt-1">Добавьте первый слайд</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="p-2 space-y-2">
        {slides.map((slide, index) => (
          <div
            key={slide.id}
            draggable
            onDragStart={() => handleDragStart(index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDrop={() => handleDrop(index)}
            onClick={() => onSelectSlide(slide.id)}
            className={`
              group relative rounded-lg border transition-all cursor-pointer
              ${selectedSlideId === slide.id
                ? 'bg-blue-50 border-blue-200 shadow-sm'
                : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'}
            `}
          >
            <div className="flex items-start gap-2 p-3">
              {/* Drag Handle */}
              <div className="flex-shrink-0 mt-1 cursor-grab active:cursor-grabbing opacity-0 group-hover:opacity-100 transition-opacity">
                <GripVertical className="w-4 h-4 text-gray-400" />
              </div>

              {/* Slide Number */}
              <div className={`
                flex-shrink-0 w-6 h-6 rounded flex items-center justify-center text-xs
                ${selectedSlideId === slide.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600'}
              `}>
                {index + 1}
              </div>

              {/* Slide Content */}
              <div className="flex-1 min-w-0">
                <h4 className={`text-sm mb-1 truncate ${
                  selectedSlideId === slide.id ? 'text-blue-900' : 'text-gray-900'
                }`}>
                  {slide.title}
                </h4>
                
                {slide.prompt ? (
                  <p className="text-xs text-gray-500 truncate mb-1">
                    {slide.prompt}
                  </p>
                ) : (
                  <p className="text-xs text-gray-400 italic mb-1">
                    Промпт не задан
                  </p>
                )}

                {/* Status indicators */}
                <div className="flex items-center gap-2 flex-wrap">
                  {slide.documents.length > 0 && (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-gray-100 rounded text-xs text-gray-600">
                      <FileText className="w-3 h-3" />
                      {slide.documents.length}
                    </span>
                  )}
                  {slide.isGenerating && (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-blue-100 rounded text-xs text-blue-600">
                      <Sparkles className="w-3 h-3 animate-pulse" />
                      Генерация...
                    </span>
                  )}
                  {slide.status === 'completed' && !slide.isGenerating && (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-green-100 rounded text-xs text-green-600">
                      ✓ Готов
                    </span>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 hover:bg-blue-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDuplicateSlide(slide.id);
                  }}
                  title="Дублировать слайд"
                >
                  <Copy className="w-3.5 h-3.5 text-blue-600" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 hover:bg-red-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSlideToDelete(slide);
                  }}
                  title="Удалить слайд"
                >
                  <Trash2 className="w-3.5 h-3.5 text-red-600" />
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Delete Slide Confirmation Dialog */}
      {slideToDelete && (
        <AlertDialog open={true} onOpenChange={() => setSlideToDelete(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Удалить слайд</AlertDialogTitle>
              <AlertDialogDescription>
                Вы уверены, что хотите удалить слайд "{slideToDelete.title}"?
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Отмена</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => {
                  onDeleteSlide(slideToDelete.id);
                  setSlideToDelete(null);
                }}
              >
                Удалить
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
}