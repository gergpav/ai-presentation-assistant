import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Button } from './ui/button';
import { Download, FileText, Loader2, CheckCircle2 } from 'lucide-react';

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  presentationTitle: string;
  slideCount: number;
  onExport: (format: 'pptx' | 'pdf') => void;
}

export function ExportDialog({
  open,
  onOpenChange,
  presentationTitle,
  slideCount,
  onExport
}: ExportDialogProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [exportComplete, setExportComplete] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<'pptx' | 'pdf' | null>(null);

  const handleExport = async (format: 'pptx' | 'pdf') => {
    setSelectedFormat(format);
    setIsExporting(true);
    setExportComplete(false);

    // Симуляция процесса экспорта
    await new Promise(resolve => setTimeout(resolve, 2000));

    onExport(format);
    setIsExporting(false);
    setExportComplete(true);

    // Автоматически закрыть диалог через 1.5 секунды
    setTimeout(() => {
      onOpenChange(false);
      setExportComplete(false);
      setSelectedFormat(null);
    }, 1500);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Экспорт презентации</DialogTitle>
          <DialogDescription>
            Выберите формат для экспорта "{presentationTitle}"
          </DialogDescription>
        </DialogHeader>

        {exportComplete ? (
          <div className="py-8 text-center">
            <CheckCircle2 className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg mb-2 text-gray-900">Экспорт завершен!</h3>
            <p className="text-sm text-gray-600">
              Файл сохранен в папку загрузок
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Слайдов в презентации:</span>
                <span className="text-gray-900">{slideCount}</span>
              </div>
            </div>

            <button
              onClick={() => handleExport('pptx')}
              disabled={isExporting}
              className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  {isExporting && selectedFormat === 'pptx' ? (
                    <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                  ) : (
                    <FileText className="w-6 h-6 text-blue-600" />
                  )}
                </div>
                <div className="flex-1">
                  <h4 className="text-gray-900 mb-1">PowerPoint (PPTX)</h4>
                  <p className="text-sm text-gray-600">
                    Редактируемый формат для PowerPoint и Google Slides
                  </p>
                </div>
                {isExporting && selectedFormat === 'pptx' && (
                  <div className="text-sm text-blue-600">Экспорт...</div>
                )}
              </div>
            </button>

            <button
              onClick={() => handleExport('pdf')}
              disabled={isExporting}
              className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-red-500 hover:bg-red-50 transition-all text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  {isExporting && selectedFormat === 'pdf' ? (
                    <Loader2 className="w-6 h-6 text-red-600 animate-spin" />
                  ) : (
                    <Download className="w-6 h-6 text-red-600" />
                  )}
                </div>
                <div className="flex-1">
                  <h4 className="text-gray-900 mb-1">PDF документ</h4>
                  <p className="text-sm text-gray-600">
                    Универсальный формат для просмотра и печати
                  </p>
                </div>
                {isExporting && selectedFormat === 'pdf' && (
                  <div className="text-sm text-red-600">Экспорт...</div>
                )}
              </div>
            </button>

            <div className="pt-4">
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={isExporting}
                className="w-full"
              >
                Отмена
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}