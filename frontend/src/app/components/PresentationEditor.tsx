import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Switch } from './ui/switch';
import { ChevronRight, Edit2, Eye, BarChart3, Image, Table, FileText } from 'lucide-react';
import type { PresentationSection, Slide, AudienceType } from '../App';

interface PresentationEditorProps {
  sections: PresentationSection[];
  audience: AudienceType;
  onSectionToggle: (sectionId: string) => void;
  onSlideEdit: (sectionId: string, slideId: string, newContent: string) => void;
}

export function PresentationEditor({
  sections,
  audience,
  onSectionToggle,
  onSlideEdit
}: PresentationEditorProps) {
  const [selectedSection, setSelectedSection] = useState<string>(sections[0]?.id || '');
  const [selectedSlide, setSelectedSlide] = useState<string>(sections[0]?.slides[0]?.id || '');
  const [editingSlide, setEditingSlide] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');

  const currentSection = sections.find(s => s.id === selectedSection);
  const currentSlide = currentSection?.slides.find(s => s.id === selectedSlide);

  const handleEditStart = (slide: Slide) => {
    setEditingSlide(slide.id);
    setEditContent(slide.content);
  };

  const handleEditSave = () => {
    if (editingSlide && currentSection) {
      onSlideEdit(currentSection.id, editingSlide, editContent);
      setEditingSlide(null);
    }
  };

  const handleEditCancel = () => {
    setEditingSlide(null);
    setEditContent('');
  };

  const getVisualIcon = (type?: string) => {
    switch (type) {
      case 'chart': return BarChart3;
      case 'image': return Image;
      case 'table': return Table;
      default: return FileText;
    }
  };

  const getAudienceLabel = (audienceType: AudienceType) => {
    const labels = {
      executive: 'Топ-менеджмент',
      expert: 'Эксперты',
      investor: 'Инвесторы'
    };
    return labels[audienceType];
  };

  const includedSections = sections.filter(s => s.included);
  const totalSlides = includedSections.reduce((acc, section) => acc + section.slides.length, 0);

  return (
    <div className="flex h-[600px]">
      {/* Left Sidebar - Sections */}
      <div className="w-80 border-r border-gray-200 bg-gray-50 overflow-y-auto">
        <div className="p-4 border-b border-gray-200 bg-white sticky top-0 z-10">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-900">Разделы презентации</h3>
            <span className="text-sm text-gray-500">{includedSections.length}/{sections.length}</span>
          </div>
          <div className="text-sm text-gray-600">
            Аудитория: <span className="font-medium text-blue-600">{getAudienceLabel(audience)}</span>
          </div>
        </div>

        <div className="p-2">
          {sections.map((section, index) => (
            <div key={section.id} className="mb-2">
              <div className={`
                rounded-lg border transition-all
                ${selectedSection === section.id 
                  ? 'bg-white border-blue-200 shadow-sm' 
                  : 'bg-white border-gray-200 hover:border-gray-300'}
              `}>
                <div className="p-3">
                  <div className="flex items-start gap-3 mb-2">
                    <Switch
                      checked={section.included}
                      onCheckedChange={() => onSectionToggle(section.id)}
                      className="mt-1"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-gray-500">
                          {index + 1}
                        </span>
                        <h4 className={`text-sm ${section.included ? 'text-gray-900' : 'text-gray-400'}`}>
                          {section.title}
                        </h4>
                      </div>
                      <p className={`text-xs ${section.included ? 'text-gray-600' : 'text-gray-400'}`}>
                        {section.description}
                      </p>
                    </div>
                  </div>

                  {section.included && (
                    <div className="ml-8 space-y-1">
                      {section.slides.map((slide) => {
                        const VisualIcon = getVisualIcon(slide.visualType);
                        return (
                          <button
                            key={slide.id}
                            onClick={() => {
                              setSelectedSection(section.id);
                              setSelectedSlide(slide.id);
                            }}
                            className={`
                              w-full text-left px-2 py-1.5 rounded text-xs flex items-center gap-2
                              ${selectedSlide === slide.id 
                                ? 'bg-blue-100 text-blue-900' 
                                : 'hover:bg-gray-100 text-gray-700'}
                            `}
                          >
                            <VisualIcon className="w-3 h-3 flex-shrink-0" />
                            <span className="truncate">{slide.title}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-gray-200 bg-white sticky bottom-0">
          <div className="text-sm text-gray-600">
            Всего слайдов: <span className="font-medium text-gray-900">{totalSlides}</span>
          </div>
        </div>
      </div>

      {/* Main Content - Slide Preview */}
      <div className="flex-1 overflow-y-auto">
        {currentSlide && currentSection ? (
          <div className="p-6">
            <Tabs defaultValue="preview">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <span>{currentSection.title}</span>
                  <ChevronRight className="w-4 h-4" />
                  <span className="text-gray-900">{currentSlide.title}</span>
                </div>
                <TabsList>
                  <TabsTrigger value="preview" className="gap-2">
                    <Eye className="w-4 h-4" />
                    Превью
                  </TabsTrigger>
                  <TabsTrigger value="edit" className="gap-2">
                    <Edit2 className="w-4 h-4" />
                    Редактор
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="preview" className="mt-0">
                <div className="bg-white rounded-lg border border-gray-200 p-8">
                  <div className="aspect-video bg-gradient-to-br from-blue-50 to-white border border-gray-200 rounded-lg p-8 flex flex-col">
                    <div className="mb-6">
                      <h2 className="text-2xl text-gray-900 mb-2">{currentSlide.title}</h2>
                      <div className="w-16 h-1 bg-blue-600 rounded"></div>
                    </div>

                    <div className="flex-1 flex items-center justify-center">
                      <div className="w-full max-w-2xl">
                        {currentSlide.visualType === 'chart' && (
                          <div className="bg-white rounded-lg p-6 border border-gray-200">
                            <div className="flex items-end justify-around h-48">
                              {[65, 85, 45, 92].map((height, i) => (
                                <div key={i} className="flex flex-col items-center gap-2">
                                  <div 
                                    className="w-16 bg-blue-500 rounded-t"
                                    style={{ height: `${height}%` }}
                                  />
                                  <span className="text-xs text-gray-600">Q{i + 1}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {currentSlide.visualType === 'table' && (
                          <div className="bg-white rounded-lg overflow-hidden border border-gray-200">
                            <table className="w-full">
                              <tbody>
                                {currentSlide.content.split('\n').map((line, i) => (
                                  <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                                    <td className="px-4 py-3 text-sm text-gray-900 border-b border-gray-200">
                                      {line}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {(!currentSlide.visualType || currentSlide.visualType === 'text') && (
                          <div className="space-y-3">
                            {currentSlide.content.split('\n').map((line, i) => (
                              <div key={i} className="flex items-start gap-3">
                                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0" />
                                <p className="text-gray-900">{line}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="mt-6 pt-4 border-t border-gray-200">
                      <div className="flex justify-between items-center text-sm text-gray-500">
                        <span>{currentSection.title}</span>
                        <span>Слайд {currentSection.slides.indexOf(currentSlide) + 1} из {currentSection.slides.length}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="edit" className="mt-0">
                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  <div className="mb-4">
                    <label className="block text-sm mb-2 text-gray-700">
                      Заголовок слайда
                    </label>
                    <div className="px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg">
                      <p className="text-gray-900">{currentSlide.title}</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm mb-2 text-gray-700">
                      Тип визуализации
                    </label>
                    <div className="flex gap-2">
                      {[
                        { type: 'text', icon: FileText, label: 'Текст' },
                        { type: 'chart', icon: BarChart3, label: 'График' },
                        { type: 'table', icon: Table, label: 'Таблица' },
                        { type: 'image', icon: Image, label: 'Изображение' }
                      ].map(({ type, icon: Icon, label }) => (
                        <div
                          key={type}
                          className={`
                            px-3 py-2 rounded-lg border flex items-center gap-2 text-sm
                            ${currentSlide.visualType === type || (!currentSlide.visualType && type === 'text')
                              ? 'bg-blue-50 border-blue-200 text-blue-900'
                              : 'bg-gray-50 border-gray-200 text-gray-600'}
                          `}
                        >
                          <Icon className="w-4 h-4" />
                          {label}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm mb-2 text-gray-700">
                      Содержание слайда
                    </label>
                    {editingSlide === currentSlide.id ? (
                      <div>
                        <Textarea
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          rows={10}
                          className="mb-3"
                          placeholder="Введите содержание слайда..."
                        />
                        <div className="flex gap-2">
                          <Button onClick={handleEditSave} size="sm">
                            Сохранить
                          </Button>
                          <Button onClick={handleEditCancel} variant="outline" size="sm">
                            Отмена
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="px-4 py-3 bg-gray-50 border border-gray-200 rounded-lg mb-3 whitespace-pre-wrap">
                          {currentSlide.content}
                        </div>
                        <Button onClick={() => handleEditStart(currentSlide)} variant="outline" size="sm">
                          <Edit2 className="w-4 h-4 mr-2" />
                          Редактировать
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>Выберите слайд для просмотра</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
