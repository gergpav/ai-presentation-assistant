import { useState } from 'react';
import { Button } from './ui/button';
import { Download, FileText, File } from 'lucide-react';
import type { PresentationSection } from '../App';

interface ExportPanelProps {
  sections: PresentationSection[];
  onExport: (format: 'pptx' | 'pdf') => void;
}

export function ExportPanel({ sections, onExport }: ExportPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<'pptx' | 'pdf'>('pptx');
  const [isExporting, setIsExporting] = useState(false);

  const totalSlides = sections.reduce((acc, section) => acc + section.slides.length, 0);

  const handleExport = async () => {
    setIsExporting(true);
    
    // Симуляция экспорта
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Создание фиктивного файла для демонстрации
    const blob = new Blob(['Презентация (демо)'], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `presentation.${selectedFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    setIsExporting(false);
    onExport(selectedFormat);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-lg mb-2 text-gray-900">Экспорт презентации</h3>
          <p className="text-sm text-gray-600 mb-6">
            Выберите формат для сохранения финальной презентации
          </p>

          <div className="grid grid-cols-2 gap-4 max-w-md mb-6">
            <button
              onClick={() => setSelectedFormat('pptx')}
              className={`
                p-4 rounded-lg border-2 transition-all text-left
                ${selectedFormat === 'pptx'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'}
              `}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  w-10 h-10 rounded-lg flex items-center justify-center
                  ${selectedFormat === 'pptx' ? 'bg-blue-100' : 'bg-gray-100'}
                `}>
                  <FileText className={`w-5 h-5 ${selectedFormat === 'pptx' ? 'text-blue-600' : 'text-gray-600'}`} />
                </div>
                <div>
                  <h4 className={`text-sm mb-1 ${selectedFormat === 'pptx' ? 'text-blue-900' : 'text-gray-900'}`}>
                    PowerPoint
                  </h4>
                  <p className="text-xs text-gray-600">Формат .pptx</p>
                  <p className="text-xs text-gray-500 mt-1">Редактируемый</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => setSelectedFormat('pdf')}
              className={`
                p-4 rounded-lg border-2 transition-all text-left
                ${selectedFormat === 'pdf'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'}
              `}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  w-10 h-10 rounded-lg flex items-center justify-center
                  ${selectedFormat === 'pdf' ? 'bg-blue-100' : 'bg-gray-100'}
                `}>
                  <File className={`w-5 h-5 ${selectedFormat === 'pdf' ? 'text-blue-600' : 'text-gray-600'}`} />
                </div>
                <div>
                  <h4 className={`text-sm mb-1 ${selectedFormat === 'pdf' ? 'text-blue-900' : 'text-gray-900'}`}>
                    PDF
                  </h4>
                  <p className="text-xs text-gray-600">Формат .pdf</p>
                  <p className="text-xs text-gray-500 mt-1">Для просмотра</p>
                </div>
              </div>
            </button>
          </div>

          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <h4 className="text-sm mb-3 text-gray-900">Сводка презентации</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-600 mb-1">Разделов</p>
                <p className="text-gray-900">{sections.length}</p>
              </div>
              <div>
                <p className="text-gray-600 mb-1">Слайдов</p>
                <p className="text-gray-900">{totalSlides}</p>
              </div>
              <div>
                <p className="text-gray-600 mb-1">Формат</p>
                <p className="text-gray-900 uppercase">{selectedFormat}</p>
              </div>
            </div>
          </div>

          <Button
            onClick={handleExport}
            disabled={isExporting}
            size="lg"
            className="gap-2"
          >
            {isExporting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Экспорт...
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                Скачать презентацию
              </>
            )}
          </Button>
        </div>

        <div className="ml-6 p-4 bg-blue-50 rounded-lg border border-blue-200 max-w-xs">
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="text-sm">
              <h4 className="mb-1 text-blue-900">Совет</h4>
              <p className="text-blue-800">
                PowerPoint формат позволит вам дополнительно редактировать презентацию в MS Office или Google Slides.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
