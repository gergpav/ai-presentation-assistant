import { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  Sparkles,
  Upload,
  FileText,
  X,
  BarChart3,
  Image,
  Table as TableIcon,
  Type,
  Loader2
} from 'lucide-react';
import type { Slide, SlideDocument } from '../../lib/types';

interface SlideEditorProps {
  slide: Slide;
  onUpdateSlide: (updates: Partial<Slide>) => void;
  onGenerateSlide: () => void;
  onUploadDocument?: (file: File) => Promise<void>;
  onDeleteDocument?: (docId: number) => Promise<void>;
}

export function SlideEditor({ slide, onUpdateSlide, onGenerateSlide, onUploadDocument, onDeleteDocument }: SlideEditorProps) {
  const [activeTab, setActiveTab] = useState<'setup' | 'content'>('setup');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      return ['xlsx', 'docx', 'pptx', 'pdf'].includes(ext || '');
    });

    if (validFiles.length === 0) return;

    if (onUploadDocument) {
      setIsUploading(true);
      try {
        for (const file of validFiles) {
          await onUploadDocument(file);
        }
        // Documents will be reloaded by parent component
      } catch (error) {
        console.error('Error uploading documents:', error);
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const handleRemoveDocument = async (docId: number) => {
    if (onDeleteDocument) {
      try {
        await onDeleteDocument(docId);
        // Document will be removed by parent component reload
      } catch (error) {
        console.error('Error deleting document:', error);
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const visualTypes = [
    { value: 'text', icon: Type, label: 'Текст' },
    { value: 'chart', icon: BarChart3, label: 'График' },
    { value: 'table', icon: TableIcon, label: 'Таблица' },
    { value: 'image', icon: Image, label: 'Изображение' }
  ] as const;

  const canGenerate = slide.prompt.trim().length > 0 && !slide.isGenerating;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
        <TabsList className="mb-6">
          <TabsTrigger value="setup">Настройка слайда</TabsTrigger>
          <TabsTrigger value="content">Контент</TabsTrigger>
        </TabsList>

        <TabsContent value="setup" className="space-y-6">
          {/* Slide Title */}
          <div>
            <Label htmlFor="slide-title">Название слайда</Label>
            <Input
              id="slide-title"
              value={slide.title || ''}
              onChange={(e) => {
                const newTitle = e.target.value;
                onUpdateSlide({ title: newTitle });
              }}
              onBlur={(e) => {
                // Сохраняем при потере фокуса
                const newTitle = e.target.value.trim();
                if (newTitle !== slide.title) {
                  onUpdateSlide({ title: newTitle });
                }
              }}
              placeholder="Например: Финансовые показатели Q4"
              className="mt-2"
            />
          </div>

          {/* Visual Type */}
          <div>
            <Label>Тип визуализации</Label>
            <div className="grid grid-cols-4 gap-3 mt-2">
              {visualTypes.map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => onUpdateSlide({ visualType: value })}
                  className={`
                    p-4 rounded-lg border-2 transition-all flex flex-col items-center gap-2
                    ${slide.visualType === value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'}
                  `}
                >
                  <Icon className={`w-6 h-6 ${
                    slide.visualType === value ? 'text-blue-600' : 'text-gray-600'
                  }`} />
                  <span className={`text-sm ${
                    slide.visualType === value ? 'text-blue-900' : 'text-gray-700'
                  }`}>
                    {label}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* AI Prompt */}
          <div>
            <Label htmlFor="prompt">AI Промпт</Label>
            <p className="text-sm text-gray-600 mt-1 mb-2">
              Опишите, что должно быть на этом слайде
            </p>
            <Textarea
              id="prompt"
              value={slide.prompt}
              onChange={(e) => onUpdateSlide({ prompt: e.target.value })}
              placeholder="Например: Создай слайд с финансовыми показателями компании за последний квартал. Покажи выручку, прибыль, маржу и ROI с процентными изменениями относительно предыдущего периода."
              rows={6}
              className="resize-none"
            />
          </div>

          {/* Documents Upload */}
          <div>
            <Label>Справочные документы</Label>
            <p className="text-sm text-gray-600 mt-1 mb-3">
              Прикрепите документы для контекста генерации (необязательно)
            </p>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 cursor-pointer transition-colors bg-gray-50"
            >
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-1">
                Нажмите для загрузки документов
              </p>
              <p className="text-xs text-gray-500">
                PPTX, DOCX, XLSX, PDF
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".xlsx,.docx,.pptx,.pdf"
              onChange={handleFileChange}
              className="hidden"
            />

            {slide.documents.length > 0 && (
              <div className="mt-4 space-y-2">
                {slide.documents.map(doc => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className="w-8 h-8 bg-blue-50 rounded flex items-center justify-center flex-shrink-0">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-gray-900 truncate">{doc.name}</p>
                        <p className="text-xs text-gray-500">{formatFileSize(doc.size)}</p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveDocument(doc.id)}
                      className="text-gray-400 hover:text-red-600 flex-shrink-0"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Generate Button */}
          <div className="pt-4 border-t border-gray-200">
            <Button
              onClick={onGenerateSlide}
              disabled={!canGenerate}
              size="lg"
              className="w-full gap-2"
            >
              {slide.isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Генерация...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Сгенерировать слайд
                </>
              )}
            </Button>

            {!slide.prompt.trim() && (
              <p className="text-sm text-gray-500 text-center mt-2">
                Введите промпт для генерации слайда
              </p>
            )}
          </div>
        </TabsContent>

        <TabsContent value="content">
          <div>
            <div className="flex items-center justify-between mb-4">
              <Label>Сгенерированный контент</Label>
              {slide.generatedContent && (
                <Button
                  onClick={onGenerateSlide}
                  disabled={!canGenerate}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Регенерировать
                </Button>
              )}
            </div>

            {slide.generatedContent ? (
              <div>
                <Textarea
                  value={slide.generatedContent}
                  onChange={(e) => onUpdateSlide({ generatedContent: e.target.value })}
                  onBlur={async (e) => {
                    // Save content when user finishes editing
                    if (onUpdateSlide && slide.id) {
                      // Content will be saved via API in parent component
                      // For now, just update local state
                    }
                  }}
                  rows={15}
                  className="resize-none font-mono text-sm"
                />
                <p className="text-sm text-gray-500 mt-2">
                  Вы можете отредактировать сгенерированный контент вручную. Изменения сохраняются автоматически.
                </p>
              </div>
            ) : (
              <div className="border-2 border-dashed border-gray-200 rounded-lg p-12 text-center">
                <Sparkles className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-gray-900 mb-2">Контент не сгенерирован</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Перейдите на вкладку "Настройка слайда" и нажмите "Сгенерировать слайд"
                </p>
                <Button
                  onClick={() => setActiveTab('setup')}
                  variant="outline"
                >
                  Перейти к настройке
                </Button>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
