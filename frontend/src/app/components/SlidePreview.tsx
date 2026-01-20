import { Eye, Sparkles } from 'lucide-react';
import type { Slide } from '../../lib/types';

interface SlidePreviewProps {
  slide: Slide;
}

export function SlidePreview({ slide }: SlidePreviewProps) {
  const renderContent = () => {
    if (!slide.generatedContent) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <Sparkles className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-sm text-gray-500">
              Контент не сгенерирован
            </p>
          </div>
        </div>
      );
    }

    if (slide.visualType === 'chart') {
      // Parse data for chart
      const lines = slide.generatedContent.split('\n').filter(l => l.trim());
      const dataLines = lines.filter(l => l.includes(':') && l.includes('₽'));
      
      return (
        <div className="p-6">
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-sm mb-4 text-gray-700">График показателей</h3>
            <div className="flex items-end justify-around h-48 gap-2">
              {dataLines.slice(0, 4).map((line, i) => {
                const height = 40 + (i * 15) + Math.random() * 20;
                const label = line.split(':')[0].trim();
                return (
                  <div key={i} className="flex flex-col items-center gap-2 flex-1">
                    <div className="w-full flex flex-col items-center">
                      <div 
                        className="w-full bg-gradient-to-t from-blue-600 to-blue-400 rounded-t"
                        style={{ height: `${height}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-600 text-center">{label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      );
    }

    if (slide.visualType === 'table') {
      const lines = slide.generatedContent.split('\n').filter(l => l.trim());
      
      return (
        <div className="p-6">
          <div className="bg-white rounded-lg overflow-hidden border border-gray-200">
            <table className="w-full">
              <tbody>
                {lines.map((line, i) => {
                  const cells = line.split('|').map(c => c.trim());
                  return (
                    <tr key={i} className={i === 0 ? 'bg-blue-50' : i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                      {cells.map((cell, j) => (
                        <td
                          key={j}
                          className={`px-3 py-2 text-xs border-b border-gray-200 ${
                            i === 0 ? 'text-blue-900' : 'text-gray-900'
                          }`}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    // Default text rendering
    const lines = slide.generatedContent.split('\n').filter(l => l.trim());
    
    return (
      <div className="p-6 space-y-2">
        {lines.map((line, i) => {
          if (line.startsWith('•')) {
            return (
              <div key={i} className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 flex-shrink-0" />
                <p className="text-sm text-gray-900">{line.substring(1).trim()}</p>
              </div>
            );
          }
          return (
            <p key={i} className="text-sm text-gray-900">{line}</p>
          );
        })}
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 text-gray-700">
          <Eye className="w-4 h-4" />
          <span className="text-sm">Превью слайда</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="bg-gradient-to-br from-blue-50 to-white rounded-lg border border-gray-200 overflow-hidden">
          {/* Slide Header */}
          <div className="bg-white border-b border-gray-200 p-4">
            <h2 className="text-lg text-gray-900">{slide.title}</h2>
            <div className="w-12 h-1 bg-blue-600 rounded mt-2"></div>
          </div>

          {/* Slide Content */}
          <div className="min-h-[300px]">
            {renderContent()}
          </div>

          {/* Slide Footer */}
          {slide.generatedContent && (
            <div className="bg-white border-t border-gray-200 p-3">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>{slide.visualType === 'text' ? 'Текстовый слайд' : 
                       slide.visualType === 'chart' ? 'График' : 
                       slide.visualType === 'table' ? 'Таблица' : 'Изображение'}</span>
                {slide.documents.length > 0 && (
                  <span>{slide.documents.length} документ(ов)</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Info */}
        {slide.prompt && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-900 mb-1">Промпт:</p>
            <p className="text-xs text-blue-700">{slide.prompt}</p>
          </div>
        )}
      </div>
    </div>
  );
}
