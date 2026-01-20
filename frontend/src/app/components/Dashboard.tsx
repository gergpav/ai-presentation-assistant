import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Label } from './ui/label';
import { ProjectCard } from './ProjectCard';
import { Plus, Search, Sparkles, LogOut, User, Upload, X, FileText } from 'lucide-react';
import type { Presentation } from '../../lib/types';

interface DashboardProps {
  projects: Presentation[];
  onCreateProject: (title: string, audience: 'executive' | 'expert' | 'investor', template?: File) => void;
  onOpenProject: (project: Presentation) => void;
  onDeleteProject: (projectId: number) => void;
  currentUser?: string | null;
  onLogout?: () => void;
  isLoading?: boolean;
}

export function Dashboard({
  projects,
  onCreateProject,
  onOpenProject,
  onDeleteProject,
  currentUser,
  onLogout,
  isLoading = false
}: DashboardProps) {
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newProjectTitle, setNewProjectTitle] = useState('');
  const [selectedAudience, setSelectedAudience] = useState<'executive' | 'expert' | 'investor'>('executive');
  const [searchQuery, setSearchQuery] = useState('');
  const [templateFile, setTemplateFile] = useState<File | null>(null);

  const handleCreate = () => {
    if (newProjectTitle.trim()) {
      onCreateProject(newProjectTitle.trim(), selectedAudience, templateFile);
      setNewProjectTitle('');
      setTemplateFile(null);
      setShowCreateDialog(false);
    }
  };

  const filteredProjects = projects.filter(project =>
    project.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const audiences = [
    { value: 'executive' as const, label: 'Топ-менеджмент', description: 'Стратегические выводы и бизнес-результаты' },
    { value: 'expert' as const, label: 'Эксперты', description: 'Детальный анализ и технические детали' },
    { value: 'investor' as const, label: 'Инвесторы', description: 'Финансовые метрики и ROI' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl text-gray-900">AI Презентации</h1>
                <p className="text-sm text-gray-600">Личный кабинет</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {currentUser && (
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-lg">
                  <User className="w-4 h-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">{currentUser}</span>
                </div>
              )}
              <Button onClick={() => setShowCreateDialog(true)} className="gap-2">
                <Plus className="w-4 h-4" />
                Новый проект
              </Button>
              {onLogout && (
                <Button onClick={onLogout} variant="outline" className="gap-2">
                  <LogOut className="w-4 h-4" />
                  Выйти
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Search Bar */}
        <div className="mb-8">
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <Input
              type="text"
              placeholder="Поиск проектов..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Projects Grid */}
        {isLoading ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-600">Загрузка проектов...</p>
          </div>
        ) : filteredProjects.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg mb-2 text-gray-900">
              {searchQuery ? 'Проекты не найдены' : 'Нет проектов'}
            </h3>
            <p className="text-gray-600 mb-6">
              {searchQuery 
                ? 'Попробуйте изменить поисковый запрос' 
                : 'Создайте свой первый проект презентации'}
            </p>
            {!searchQuery && (
              <Button onClick={() => setShowCreateDialog(true)} className="gap-2">
                <Plus className="w-4 h-4" />
                Создать проект
              </Button>
            )}
          </div>
        ) : (
          <>
            <div className="mb-4">
              <p className="text-sm text-gray-600">
                Всего проектов: <span className="font-medium text-gray-900">{filteredProjects.length}</span>
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredProjects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onOpen={() => onOpenProject(project)}
                  onDelete={() => onDeleteProject(project.id)}
                />
              ))}
            </div>
          </>
        )}
      </main>

      {/* Create Project Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Создать новый проект</DialogTitle>
            <DialogDescription>
              Введите название презентации, выберите тип аудитории и опционально загрузите шаблон PPTX
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            <div className="space-y-2">
              <Label htmlFor="title">Название презентации</Label>
              <Input
                id="title"
                placeholder="Например: Кватальный отчёт Q4 2024"
                value={newProjectTitle}
                onChange={(e) => setNewProjectTitle(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <Label>Тип аудитории</Label>
              <div className="space-y-2">
                {audiences.map((audience) => (
                  <button
                    key={audience.value}
                    onClick={() => setSelectedAudience(audience.value)}
                    className={`
                      w-full text-left p-4 rounded-lg border-2 transition-all
                      ${selectedAudience === audience.value
                        ? 'border-blue-600 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                      }
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium text-gray-900 mb-1">
                          {audience.label}
                        </div>
                        <div className="text-sm text-gray-600">
                          {audience.description}
                        </div>
                      </div>
                      {selectedAudience === audience.value && (
                        <div className="w-5 h-5 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Загрузить шаблон презентации (опционально)</Label>
              <p className="text-xs text-gray-500 mb-2">
                Загрузите PPTX файл для использования в качестве шаблона
              </p>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 hover:border-blue-400 transition-colors">
                <label htmlFor="template-upload" className="cursor-pointer">
                  <input
                    id="template-upload"
                    type="file"
                    accept=".pptx"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file && file.name.endsWith('.pptx')) {
                        setTemplateFile(file);
                      } else if (file) {
                        alert('Пожалуйста, загрузите файл в формате PPTX');
                        e.target.value = '';
                      }
                    }}
                  />
                  {templateFile ? (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                          <FileText className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900">{templateFile.name}</p>
                          <p className="text-xs text-gray-500">
                            {(templateFile.size / 1024).toFixed(1)} KB
                          </p>
                        </div>
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.preventDefault();
                          setTemplateFile(null);
                          const input = document.getElementById('template-upload') as HTMLInputElement;
                          if (input) input.value = '';
                        }}
                      >
                        <X className="w-4 h-4 text-gray-500" />
                      </Button>
                    </div>
                  ) : (
                    <div className="text-center">
                      <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600 mb-1">
                        Нажмите для загрузки шаблона
                      </p>
                      <p className="text-xs text-gray-500">
                        Только PPTX формат
                      </p>
                    </div>
                  )}
                </label>
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3">
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Отмена
            </Button>
            <Button onClick={handleCreate} disabled={!newProjectTitle.trim()}>
              Создать проект
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}