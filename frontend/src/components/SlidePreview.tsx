import { SimpleCard } from './SimpleCard';
import { SimpleButton } from './SimpleButton';
import { Trash2 } from './Icons';
import type { Slide } from './SlideEditor';

interface SlidePreviewProps {
  slide: Slide;
  isActive: boolean;
  onClick: () => void;
  onDelete?: () => void;
  index: number;
  hasError?: boolean;
}

export function SlidePreview({ slide, isActive, onClick, onDelete, index, hasError }: SlidePreviewProps) {
  return (
    <SimpleCard
      className={`p-4 transition-all hover:shadow-md ${
        isActive ? 'ring-2 ring-[#0057B8] shadow-md' : ''
      } ${hasError ? 'ring-2 ring-red-500 bg-red-50' : ''}`}
    >
      <div className="flex items-start gap-3">
        <div 
          className="flex-shrink-0 w-8 h-8 rounded bg-[#0057B8] text-white flex items-center justify-center text-sm cursor-pointer"
          onClick={onClick}
        >
          {index + 1}
        </div>
        <div className="flex-1 min-w-0 cursor-pointer" onClick={onClick}>
          <h4 className="truncate mb-2">{slide.title || 'Без названия'}</h4>
          <div className="space-y-1">
            {slide.blocks.slice(0, 3).map((block) => (
              <div key={block.id} className="text-xs text-gray-500 truncate">
                {block.type === 'text' && block.content && `📝 ${block.content.substring(0, 30)}${block.content.length > 30 ? '...' : ''}`}
                {block.type === 'text' && !block.content && '📝 Пустой текст'}
                {block.type === 'image' && '🖼️ Изображение'}
                {block.type === 'prompt' && '✨ AI Промпт'}
                {block.type === 'pdf' && block.content.name && `📄 ${block.content.name}`}
                {block.type === 'pdf' && !block.content.name && '📄 PDF'}
                {block.type === 'pptx' && block.content.name && `📊 ${block.content.name}`}
                {block.type === 'pptx' && !block.content.name && '📊 PPTX'}
                {block.type === 'docx' && block.content.name && `📝 ${block.content.name}`}
                {block.type === 'docx' && !block.content.name && '📝 DOCX'}
                {block.type === 'xlsx' && block.content.name && `📊 ${block.content.name}`}
                {block.type === 'xlsx' && !block.content.name && '📊 XLSX'}
              </div>
            ))}
            {slide.blocks.length > 3 && (
              <div className="text-xs text-gray-400">+{slide.blocks.length - 3} ещё</div>
            )}
            {slide.blocks.length === 0 && (
              <div className="text-xs text-gray-400">Нет блоков</div>
            )}
          </div>
        </div>
        {onDelete && (
          <SimpleButton
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="flex-shrink-0"
          >
            <Trash2 className="w-4 h-4 text-red-500" />
          </SimpleButton>
        )}
      </div>
    </SimpleCard>
  );
}