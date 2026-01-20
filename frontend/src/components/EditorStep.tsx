import { useState } from 'react';
import { Plus, ArrowRight, Trash2 } from './Icons';
import { SimpleButton } from './SimpleButton';
import { SimpleSelect } from './SimpleInput';
import { SimpleLabel } from './SimpleLabel';
import { SlideEditor, Slide } from './SlideEditor';
import { SlidePreview } from './SlidePreview';

interface EditorStepProps {
  onContinue: (slides: Slide[]) => void;
}

export function EditorStep({ onContinue }: EditorStepProps) {
  const [slides, setSlides] = useState<Slide[]>([
    {
      id: 'slide-1',
      title: '',
      blocks: [
        {
          id: 'block-1',
          type: 'prompt',
          content: '',
        },
      ],
    },
  ]);
  
  const [activeSlideId, setActiveSlideId] = useState(slides[0]?.id);
  const [audience, setAudience] = useState('investors');
  const [errorSlides, setErrorSlides] = useState<Set<string>>(new Set());
  const [errorBlocks, setErrorBlocks] = useState<Set<string>>(new Set());

  const activeSlide = slides.find(s => s.id === activeSlideId);

  const updateSlide = (updatedSlide: Slide) => {
    setSlides(slides.map(s => (s.id === updatedSlide.id ? updatedSlide : s)));
  };

  const addSlide = () => {
    const newSlide: Slide = {
      id: `slide-${Date.now()}`,
      title: '',
      blocks: [
        {
          id: `block-${Date.now()}`,
          type: 'prompt',
          content: '',
        },
      ],
    };
    setSlides([...slides, newSlide]);
    setActiveSlideId(newSlide.id);
  };

  const deleteSlide = (slideId: string) => {
    const newSlides = slides.filter(s => s.id !== slideId);
    if (newSlides.length === 0) {
      // Don't allow deleting the last slide
      return;
    }
    setSlides(newSlides);
    if (activeSlideId === slideId) {
      setActiveSlideId(newSlides[0].id);
    }
  };

  const hasPrompt = () => {
    return slides.some(slide => 
      slide.blocks.some(block => block.type === 'prompt' && block.content.trim() !== '')
    );
  };

  const validateSlides = () => {
    const slideErrors = new Set<string>();
    const blockErrors = new Set<string>();
    
    slides.forEach((slide) => {
      let hasError = false;
      
      // Проверка заголовка слайда
      if (!slide.title.trim()) {
        hasError = true;
      }
      
      // Проверка AI промпта
      const promptBlock = slide.blocks.find(block => block.type === 'prompt');
      if (promptBlock && !promptBlock.content.trim()) {
        hasError = true;
        blockErrors.add(promptBlock.id);
      }
      
      // Проверка документов
      slide.blocks.forEach(block => {
        if ((block.type === 'pdf' || block.type === 'pptx' || block.type === 'xlsx' || block.type === 'docx') && !block.content.name) {
          hasError = true;
          blockErrors.add(block.id);
        }
      });
      
      if (hasError) {
        slideErrors.add(slide.id);
      }
    });
    
    return { slideErrors, blockErrors };
  };

  const handleContinue = () => {
    const { slideErrors, blockErrors } = validateSlides();
    if (slideErrors.size > 0) {
      setErrorSlides(slideErrors);
      setErrorBlocks(blockErrors);
      return;
    }
    onContinue(slides);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <h2>Редактор презентации</h2>
          <div className="flex items-center gap-3">
            <SimpleLabel className="text-sm">Аудитория:</SimpleLabel>
            <SimpleSelect 
              value={audience} 
              onChange={(e) => setAudience(e.target.value)}
              className="w-48"
            >
              <option value="investors">Инвесторы</option>
              <option value="executives">Топ-менеджеры</option>
              <option value="experts">Эксперты</option>
            </SimpleSelect>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - Slide List */}
        <div className="w-80 flex flex-col bg-white border-r">
          <div className="p-4 border-b">
            <h3 className="mb-3">Структура ({slides.length})</h3>
            <SimpleButton onClick={addSlide} className="w-full" variant="outline">
              <Plus className="w-4 h-4 mr-2" />
              Добавить слайд
            </SimpleButton>
          </div>
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 space-y-2">
              {slides.map((slide, index) => (
                <SlidePreview
                  key={slide.id}
                  slide={slide}
                  index={index}
                  isActive={slide.id === activeSlideId}
                  onClick={() => setActiveSlideId(slide.id)}
                  onDelete={slides.length > 1 ? () => deleteSlide(slide.id) : undefined}
                  hasError={errorSlides.has(slide.id)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Editor Panel */}
        <div className="flex-1 bg-gray-50">
          {activeSlide ? (
            <SlideEditor slide={activeSlide} onUpdate={updateSlide} errorBlocks={errorBlocks} />
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              Выберите слайд для редактирования
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="bg-white border-t px-6 py-4">
        <div className="flex items-center justify-end gap-2">
          <span className="text-sm text-gray-500">
            {slides.length} {slides.length === 1 ? 'слайд' : 'слайдов'}
          </span>
          <SimpleButton
            onClick={handleContinue}
            disabled={!hasPrompt()}
          >
            Экспорт
            <ArrowRight className="w-4 h-4 ml-2" />
          </SimpleButton>
        </div>
      </div>
    </div>
  );
}