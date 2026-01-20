import { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Upload, FileText, Trash2, CheckCircle } from 'lucide-react';
import type { UploadedFile } from '../App';

interface FileUploadSectionProps {
  onFilesUpload: (files: UploadedFile[]) => void;
}

export function FileUploadSection({ onFilesUpload }: FileUploadSectionProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    addFiles(selectedFiles);
  };

  const addFiles = (selectedFiles: File[]) => {
    const validFiles = selectedFiles.filter(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      return ['xlsx', 'docx', 'pptx', 'pdf'].includes(ext || '');
    });

    const newFiles: UploadedFile[] = validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      type: file.type,
      size: file.size,
      file
    }));

    setFiles([...files, ...newFiles]);
  };

  const removeFile = (fileId: string) => {
    setFiles(files.filter(f => f.id !== fileId));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleContinue = () => {
    if (files.length > 0) {
      onFilesUpload(files);
    }
  };

  return (
    <div className="p-8">
      <div className="max-w-3xl mx-auto">
        <div className="mb-8 text-center">
          <h2 className="text-2xl mb-2 text-gray-900">Загрузите документы</h2>
          <p className="text-gray-600">
            Поддерживаемые форматы: XLSX, DOCX, PPTX, PDF
          </p>
        </div>

        {/* Upload Area */}
        <div
          className={`
            border-2 border-dashed rounded-lg p-12 text-center transition-colors
            ${isDragging 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 bg-gray-50 hover:bg-gray-100'}
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
              <Upload className="w-8 h-8 text-blue-600" />
            </div>
            <p className="mb-2 text-gray-900">
              Перетащите файлы сюда или
            </p>
            <Button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
            >
              Выберите файлы
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".xlsx,.docx,.pptx,.pdf"
              onChange={handleFileChange}
              className="hidden"
            />
            <p className="text-sm text-gray-500 mt-4">
              Максимальный размер файла: 50 МБ
            </p>
          </div>
        </div>

        {/* Files List */}
        {files.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-gray-900">Загруженные файлы ({files.length})</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFiles([])}
                className="text-gray-500"
              >
                Очистить все
              </Button>
            </div>
            
            <div className="space-y-2">
              {files.map(file => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className="w-10 h-10 bg-blue-50 rounded flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-900 truncate">{file.name}</p>
                      <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                    </div>
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(file.id)}
                    className="ml-4 text-gray-400 hover:text-red-600"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>

            <div className="mt-6 flex justify-center">
              <Button 
                onClick={handleContinue}
                size="lg"
                className="px-8"
              >
                Продолжить
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
