import { useState } from 'react';
import { Image, Type, Trash2, Wand2, FilePdf, FilePresentation, FileSpreadsheet, FileText, Upload } from './Icons';
import { SimpleButton } from './SimpleButton';
import { SimpleInput, SimpleTextarea } from './SimpleInput';
import { SimpleLabel } from './SimpleLabel';
import { SimpleCard } from './SimpleCard';
import { ImageWithFallback } from './figma/ImageWithFallback';

export interface SlideBlock {
  id: string;
  type: 'text' | 'image' | 'prompt' | 'pdf' | 'pptx' | 'xlsx' | 'docx';
  content: any;
}

export interface Slide {
  id: string;
  title: string;
  blocks: SlideBlock[];
}

interface SlideEditorProps {
  slide: Slide;
  onUpdate: (slide: Slide) => void;
  errorBlocks?: Set<string>;
}

export function SlideEditor({ slide, onUpdate, errorBlocks }: SlideEditorProps) {
  const [showImageDialog, setShowImageDialog] = useState(false);
  const [currentBlockId, setCurrentBlockId] = useState('');
  const [imageSearchQuery, setImageSearchQuery] = useState('');
  const [loadedPrompts, setLoadedPrompts] = useState<Set<string>>(new Set());

  const updateTitle = (title: string) => {
    onUpdate({ ...slide, title });
  };

  const addBlock = (type: SlideBlock['type']) => {
    const newBlock: SlideBlock = {
      id: `block-${Date.now()}`,
      type,
      content: getDefaultContent(type),
    };
    onUpdate({ ...slide, blocks: [...slide.blocks, newBlock] });
  };

  const updateBlock = (blockId: string, content: any) => {
    onUpdate({
      ...slide,
      blocks: slide.blocks.map(block =>
        block.id === blockId ? { ...block, content } : block
      ),
    });
  };

  const deleteBlock = (blockId: string) => {
    onUpdate({
      ...slide,
      blocks: slide.blocks.filter(block => block.id !== blockId),
    });
  };

  const getDefaultContent = (type: SlideBlock['type']) => {
    switch (type) {
      case 'text':
        return '';
      case 'image':
        return { url: '', alt: '' };
      case 'prompt':
        return 'Опишите, что должен сгенерировать AI...';
      case 'pdf':
        return { file: null, name: '' };
      case 'pptx':
        return { file: null, name: '' };
      case 'xlsx':
        return { file: null, name: '' };
      case 'docx':
        return { file: null, name: '' };
      default:
        return '';
    }
  };

  const openImageDialog = (blockId: string) => {
    setCurrentBlockId(blockId);
    setShowImageDialog(true);
  };

  const applyImage = () => {
    updateBlock(currentBlockId, {
      url: `https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800`,
      alt: imageSearchQuery || 'Presentation image',
    });
    setShowImageDialog(false);
    setImageSearchQuery('');
  };

  const handleLoadPrompt = (blockId: string) => {
    setLoadedPrompts(prev => new Set(prev).add(blockId));
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b bg-white">
        <SimpleLabel>Заголовок слайда</SimpleLabel>
        <SimpleInput
          value={slide.title}
          onChange={(e) => updateTitle(e.target.value)}
          placeholder="Введите заголовок..."
          className="mt-2"
        />
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {slide.blocks.map((block) => {
          const hasError = errorBlocks?.has(block.id);
          return (
          <SimpleCard key={block.id} className={`p-4 ${hasError ? 'ring-2 ring-red-500 bg-red-50' : ''}`}>
            <div className="flex items-center justify-between mb-3">
              <SimpleLabel className="flex items-center gap-2">
                {block.type === 'text' && <Type className="w-4 h-4" />}
                {block.type === 'image' && <Image className="w-4 h-4" />}
                {block.type === 'prompt' && <Wand2 className="w-4 h-4" />}
                {block.type === 'pdf' && <FilePdf className="w-4 h-4" />}
                {block.type === 'pptx' && <FilePresentation className="w-4 h-4" />}
                {block.type === 'xlsx' && <FileSpreadsheet className="w-4 h-4" />}
                {block.type === 'docx' && <FileText className="w-4 h-4" />}
                {block.type === 'text' && 'Текст'}
                {block.type === 'image' && 'Изображение'}
                {block.type === 'prompt' && 'AI Промпт'}
                {block.type === 'pdf' && 'PDF документ'}
                {block.type === 'pptx' && 'PPTX презентация'}
                {block.type === 'xlsx' && 'XLSX таблица'}
                {block.type === 'docx' && 'DOCX документ'}
              </SimpleLabel>
              {block.type !== 'prompt' && (
                <SimpleButton
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteBlock(block.id)}
                >
                  <Trash2 className="w-4 h-4 text-red-500" />
                </SimpleButton>
              )}
            </div>

            {block.type === 'text' && (
              <SimpleTextarea
                value={block.content}
                onChange={(e) => updateBlock(block.id, e.target.value)}
                placeholder="Введите текст..."
                rows={4}
              />
            )}

            {block.type === 'prompt' && (
              <div className="space-y-3">
                <SimpleTextarea
                  value={block.content}
                  onChange={(e) => updateBlock(block.id, e.target.value)}
                  placeholder="Опишите, чо должен сгенерировать AI для этого слайда..."
                  rows={4}
                  className="bg-purple-50 border-purple-200"
                />
                <SimpleButton 
                  variant="outline" 
                  size="sm" 
                  className="w-full"
                  onClick={() => handleLoadPrompt(block.id)}
                  disabled={!block.content.trim()}
                >
                  <Wand2 className="w-4 h-4 mr-2" />
                  Загрузить промпт
                </SimpleButton>
                {loadedPrompts.has(block.id) && (
                  <p className="text-xs text-green-600">
                    ✓ Ваш промпт загружен
                  </p>
                )}
              </div>
            )}

            {block.type === 'pdf' && (
              <div className="space-y-3">
                <SimpleButton 
                  variant="outline" 
                  className="w-full"
                  onClick={() => document.getElementById(`pdf-upload-${block.id}`)?.click()}
                >
                  <FilePdf className="w-4 h-4 mr-2" />
                  {block.content.name ? 'Изменить PDF' : 'Загрузить PDF'}
                </SimpleButton>
                <input
                  id={`pdf-upload-${block.id}`}
                  type="file"
                  accept=".pdf"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      updateBlock(block.id, {
                        file: e.target.files[0],
                        name: e.target.files[0].name,
                      });
                    }
                  }}
                  className="hidden"
                />
                {block.content.name && (
                  <div className="p-3 bg-gray-50 rounded border flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FilePdf className="w-5 h-5 text-red-500" />
                      <span className="text-sm">{block.content.name}</span>
                    </div>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  PDF будет встроен в презентацию или экспортирован как приложение
                </p>
              </div>
            )}

            {block.type === 'pptx' && (
              <div className="space-y-3">
                <SimpleButton 
                  variant="outline" 
                  className="w-full"
                  onClick={() => document.getElementById(`pptx-upload-${block.id}`)?.click()}
                >
                  <FilePresentation className="w-4 h-4 mr-2" />
                  {block.content.name ? 'Изменить PPTX' : 'Загрузить PPTX'}
                </SimpleButton>
                <input
                  id={`pptx-upload-${block.id}`}
                  type="file"
                  accept=".pptx"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      updateBlock(block.id, {
                        file: e.target.files[0],
                        name: e.target.files[0].name,
                      });
                    }
                  }}
                  className="hidden"
                />
                {block.content.name && (
                  <div className="p-3 bg-gray-50 rounded border flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FilePresentation className="w-5 h-5 text-orange-500" />
                      <span className="text-sm">{block.content.name}</span>
                    </div>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  Слайды из PPTX будут импортированы в вашу презентацию
                </p>
              </div>
            )}

            {block.type === 'xlsx' && (
              <div className="space-y-3">
                <SimpleButton 
                  variant="outline" 
                  className="w-full"
                  onClick={() => document.getElementById(`xlsx-upload-${block.id}`)?.click()}
                >
                  <FileSpreadsheet className="w-4 h-4 mr-2" />
                  {block.content.name ? 'Изменить XLSX' : 'Загрузить XLSX'}
                </SimpleButton>
                <input
                  id={`xlsx-upload-${block.id}`}
                  type="file"
                  accept=".xlsx,.xls"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      updateBlock(block.id, {
                        file: e.target.files[0],
                        name: e.target.files[0].name,
                      });
                    }
                  }}
                  className="hidden"
                />
                {block.content.name && (
                  <div className="p-3 bg-gray-50 rounded border flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileSpreadsheet className="w-5 h-5 text-green-500" />
                      <span className="text-sm">{block.content.name}</span>
                    </div>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  Данные из XLSX будут использованы для генерации контента
                </p>
              </div>
            )}

            {block.type === 'docx' && (
              <div className="space-y-3">
                <SimpleButton 
                  variant="outline" 
                  className="w-full"
                  onClick={() => document.getElementById(`docx-upload-${block.id}`)?.click()}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  {block.content.name ? 'Изменить DOCX' : 'Загрузить DOCX'}
                </SimpleButton>
                <input
                  id={`docx-upload-${block.id}`}
                  type="file"
                  accept=".docx,.doc"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      updateBlock(block.id, {
                        file: e.target.files[0],
                        name: e.target.files[0].name,
                      });
                    }
                  }}
                  className="hidden"
                />
                {block.content.name && (
                  <div className="p-3 bg-gray-50 rounded border flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-500" />
                      <span className="text-sm">{block.content.name}</span>
                    </div>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  Текст из DOCX будет использован для генерации контента
                </p>
              </div>
            )}

            {block.type === 'image' && (
              <div className="space-y-3">
                <div className="flex gap-2">
                  <SimpleButton 
                    variant="outline" 
                    className="flex-1"
                    onClick={() => openImageDialog(block.id)}
                  >
                    <Image className="w-4 h-4 mr-2" />
                    Подобрать
                  </SimpleButton>
                  <SimpleButton 
                    variant="outline" 
                    className="flex-1"
                    onClick={() => document.getElementById(`image-upload-${block.id}`)?.click()}
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Загрузить
                  </SimpleButton>
                </div>
                <input
                  id={`image-upload-${block.id}`}
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      const file = e.target.files[0];
                      const reader = new FileReader();
                      reader.onload = (event) => {
                        updateBlock(block.id, {
                          url: event.target?.result,
                          alt: file.name,
                        });
                      };
                      reader.readAsDataURL(file);
                    }
                  }}
                  className="hidden"
                />
                {block.content.url && (
                  <div className="relative">
                    <ImageWithFallback
                      src={block.content.url}
                      alt={block.content.alt}
                      className="w-full h-48 object-cover rounded"
                    />
                  </div>
                )}
              </div>
            )}
          </SimpleCard>
        );
        })}
      </div>

      {/* Add Block Buttons */}
      <div className="border-t bg-white p-4">
        <SimpleLabel className="mb-3">Добавить блок</SimpleLabel>
        <div className="flex flex-wrap gap-2">
          <SimpleButton
            variant="outline"
            size="sm"
            onClick={() => addBlock('image')}
          >
            <Image className="w-4 h-4 mr-2" />
            Изображение
          </SimpleButton>
          <SimpleButton
            variant="outline"
            size="sm"
            onClick={() => addBlock('pdf')}
          >
            <FilePdf className="w-4 h-4 mr-2" />
            PDF
          </SimpleButton>
          <SimpleButton
            variant="outline"
            size="sm"
            onClick={() => addBlock('pptx')}
          >
            <FilePresentation className="w-4 h-4 mr-2" />
            PPTX
          </SimpleButton>
          <SimpleButton
            variant="outline"
            size="sm"
            onClick={() => addBlock('xlsx')}
          >
            <FileSpreadsheet className="w-4 h-4 mr-2" />
            XLSX
          </SimpleButton>
          <SimpleButton
            variant="outline"
            size="sm"
            onClick={() => addBlock('docx')}
          >
            <FileText className="w-4 h-4 mr-2" />
            DOCX
          </SimpleButton>
        </div>
      </div>

      {/* Simple Dialog for Image Selection */}
      {showImageDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowImageDialog(false)}>
          <div className="bg-white rounded-lg p-6 max-w-md w-full m-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="mb-4">Подбор изображения</h3>
            <div className="space-y-4">
              <div>
                <SimpleLabel>Поисковый запрос</SimpleLabel>
                <SimpleInput
                  value={imageSearchQuery}
                  onChange={(e) => setImageSearchQuery(e.target.value)}
                  placeholder="business meeting, startup, technology..."
                  className="mt-2"
                />
              </div>
              <p className="text-sm text-gray-500">
                В реальной версии здесь будет AI-генерация изображений на основе содержимого слайда
              </p>
              <div className="flex gap-2 justify-end">
                <SimpleButton variant="outline" onClick={() => setShowImageDialog(false)}>
                  Отмена
                </SimpleButton>
                <SimpleButton onClick={applyImage}>
                  Применить
                </SimpleButton>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}