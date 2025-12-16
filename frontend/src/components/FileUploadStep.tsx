import { useState, useCallback } from 'react';
import { Upload, FileText, FileSpreadsheet, X } from './Icons';
import { SimpleButton } from './SimpleButton';
import { SimpleCard } from './SimpleCard';

interface FileUploadStepProps {
  onContinue: (files: File[]) => void;
}

const getFileIcon = (type: string) => {
  if (type.includes('word') || type.includes('document')) return FileText;
  if (type.includes('sheet') || type.includes('excel')) return FileSpreadsheet;
  if (type.includes('presentation') || type.includes('powerpoint')) return FileText;
  if (type.includes('pdf')) return FileText;
  return FileText;
};

export function FileUploadStep({ onContinue }: FileUploadStepProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(prev => [...prev, ...droppedFiles]);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...selectedFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8 bg-gray-50">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-[#0057B8] mb-2">Генератор инвестиционных презентаций</h1>
          <p className="text-gray-600">Загрузите документы для автоматической сборки презентации</p>
        </div>

        <SimpleCard className="p-8">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging 
                ? 'border-[#00A8E8] bg-[#00A8E8]/5' 
                : 'border-gray-300 hover:border-[#0057B8]'
            }`}
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-[#0057B8]" />
            <h3 className="mb-2">Перетащите файлы сюда</h3>
            <p className="text-gray-500 mb-4">или</p>
            <SimpleButton 
              variant="outline" 
              type="button"
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              Выбрать файлы
            </SimpleButton>
            <input
              id="file-upload"
              type="file"
              multiple
              accept=".docx,.xlsx,.pptx,.pdf"
              onChange={handleFileInput}
              className="hidden"
            />
            <p className="text-sm text-gray-500 mt-4">
              Поддерживаются форматы: DOCX, XLSX, PPTX, PDF
            </p>
          </div>

          {files.length > 0 && (
            <div className="mt-6">
              <h4 className="mb-3">Загруженные файлы ({files.length})</h4>
              <div className="space-y-2">
                {files.map((file, index) => {
                  const Icon = getFileIcon(file.type);
                  return (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <Icon className="w-5 h-5 text-[#0057B8]" />
                        <span>{file.name}</span>
                        <span className="text-sm text-gray-500">
                          ({(file.size / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                      <SimpleButton
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(index)}
                      >
                        <X className="w-4 h-4" />
                      </SimpleButton>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="mt-8 flex justify-end">
            <SimpleButton
              onClick={() => onContinue(files)}
              disabled={files.length === 0}
            >
              Продолжить
            </SimpleButton>
          </div>
        </SimpleCard>
      </div>
    </div>
  );
}
