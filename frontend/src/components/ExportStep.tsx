import { useState } from 'react';
import { Download, FileText, Upload, CheckCircle2, ArrowLeft, Loader2 } from './Icons';
import { SimpleButton } from './SimpleButton';
import { SimpleCard } from './SimpleCard';
import { SimpleLabel } from './SimpleLabel';
import { SimpleProgress } from './SimpleProgress';
import type { Slide } from './SlideEditor';

interface ExportStepProps {
  slides: Slide[];
  onBack: () => void;
}

export function ExportStep({ slides, onBack }: ExportStepProps) {
  const [exportFormat, setExportFormat] = useState<'pptx' | 'pdf'>('pptx');
  const [customTemplate, setCustomTemplate] = useState<File | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  const handleTemplateUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setCustomTemplate(e.target.files[0]);
    }
  };

  const handleExport = () => {
    setIsExporting(true);
    setExportProgress(0);

    const interval = setInterval(() => {
      setExportProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsExporting(false);
          setIsComplete(true);
          return 100;
        }
        return prev + 10;
      });
    }, 200);

    setTimeout(() => {
      const mockData = new Blob(['Mock presentation data'], { type: 'application/octet-stream' });
      const url = window.URL.createObjectURL(mockData);
      const link = document.createElement('a');
      link.href = url;
      link.download = `presentation.${exportFormat}`;
      window.URL.revokeObjectURL(url);
    }, 2500);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8 bg-gray-50">
      <div className="max-w-3xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-[#0057B8] mb-2">Экспорт презентации</h1>
          <p className="text-gray-600">Настройте параметры экспорта и скачайте готовый файл</p>
        </div>

        <SimpleCard className="p-8 space-y-6">
          {/* Summary */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="mb-3">Информация о презентации</h3>
            <div className="text-sm">
              <span className="text-gray-600">Количество слайдов:</span>
              <span className="ml-2 font-medium">{slides.length}</span>
            </div>
          </div>

          {/* Format Selection */}
          <div className="space-y-3">
            <SimpleLabel>Формат экспорта</SimpleLabel>
            <div className="space-y-2">
              <div 
                onClick={() => setExportFormat('pptx')}
                className={`flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer transition-all ${
                  exportFormat === 'pptx' ? 'border-[#0057B8] bg-[#0057B8]/5' : ''
                }`}
              >
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  exportFormat === 'pptx' ? 'border-[#0057B8]' : 'border-gray-300'
                }`}>
                  {exportFormat === 'pptx' && (
                    <div className="w-3 h-3 rounded-full bg-[#0057B8]" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-[#0057B8]" />
                    <div>
                      <div>PowerPoint (.pptx)</div>
                      <div className="text-sm text-gray-500">Редактируемый формат для дальнейшей работы</div>
                    </div>
                  </div>
                </div>
              </div>
              <div 
                onClick={() => setExportFormat('pdf')}
                className={`flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer transition-all ${
                  exportFormat === 'pdf' ? 'border-[#0057B8] bg-[#0057B8]/5' : ''
                }`}
              >
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  exportFormat === 'pdf' ? 'border-[#0057B8]' : 'border-gray-300'
                }`}>
                  {exportFormat === 'pdf' && (
                    <div className="w-3 h-3 rounded-full bg-[#0057B8]" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-[#0057B8]" />
                    <div>
                      <div>PDF документ (.pdf)</div>
                      <div className="text-sm text-gray-500">Универсальный формат для распространения</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Template Upload */}
          {exportFormat === 'pptx' && (
            <div className="space-y-3">
              <SimpleLabel>Шаблон оформления (опционально)</SimpleLabel>
              <div className="border-2 border-dashed rounded-lg p-6 text-center">
                {customTemplate ? (
                  <div className="flex items-center justify-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                    <span>{customTemplate.name}</span>
                    <SimpleButton
                      variant="ghost"
                      size="sm"
                      onClick={() => setCustomTemplate(null)}
                    >
                      Удалить
                    </SimpleButton>
                  </div>
                ) : (
                  <div>
                    <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                    <SimpleButton 
                      variant="outline" 
                      type="button"
                      onClick={() => document.getElementById('template-upload')?.click()}
                    >
                      Загрузить шаблон .pptx
                    </SimpleButton>
                    <input
                      id="template-upload"
                      type="file"
                      accept=".pptx"
                      onChange={handleTemplateUpload}
                      className="hidden"
                    />
                    <p className="text-sm text-gray-500 mt-2">
                      Загрузите свой корпоративный шаблон
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Export Progress */}
          {isExporting && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <SimpleLabel>Генерация презентации...</SimpleLabel>
                <span className="text-sm text-gray-600">{exportProgress}%</span>
              </div>
              <SimpleProgress value={exportProgress} />
            </div>
          )}

          {/* Success Message */}
          {isComplete && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
              <div>
                <div className="text-green-900">Презентация готова!</div>
                <div className="text-sm text-green-700">Файл будет автоматически загружен</div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-4 border-t">
            <SimpleButton variant="outline" onClick={onBack} disabled={isExporting}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Назад к редактору
            </SimpleButton>
            <SimpleButton
              onClick={isComplete ? onBack : handleExport}
              disabled={isExporting}
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Экспорт...
                </>
              ) : isComplete ? (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Готово
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Экспортировать
                </>
              )}
            </SimpleButton>
          </div>
        </SimpleCard>

        {/* Additional Info */}
        <div className="mt-6 text-center text-sm text-gray-500">
          <p>В реальной версии здесь будет интеграция с AI для генерации контента</p>
          <p>и автоматическая обработка загруженных документов</p>
        </div>
      </div>
    </div>
  );
}