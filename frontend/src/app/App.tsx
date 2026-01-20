import { useState, useEffect } from 'react';
import { SlideEditor } from './components/SlideEditor';
import { SlideList } from './components/SlideList';
import { SlidePreview } from './components/SlidePreview';
import { PresentationHeader } from './components/PresentationHeader';
import { Dashboard } from './components/Dashboard';
import { LoginScreen } from './components/LoginScreen';
import { ExportDialog } from './components/ExportDialog';
import { Button } from './components/ui/button';
import { Plus } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../lib/api';
import type { Presentation, Slide, SlideDocument } from '../lib/types';
import { projectToPresentation, projectListItemToPresentation } from '../lib/types';

export type { Presentation, Slide, SlideDocument };

export default function App() {
  const { isAuthenticated, isLoading, user, logout } = useAuth();
  const [presentation, setPresentation] = useState<Presentation | null>(null);
  const [selectedSlideId, setSelectedSlideId] = useState<number | null>(null);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [projects, setProjects] = useState<Presentation[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load projects from API when authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      setProjects([]);
      setPresentation(null);
      return;
    }

    const loadProjects = async () => {
      setIsLoadingProjects(true);
      setError(null);
      try {
        const projectList = await api.listProjects();
        const presentations = projectList.map(projectListItemToPresentation);
        setProjects(presentations);
      } catch (err: any) {
        setError(err.message || 'Ошибка загрузки проектов');
        console.error('Error loading projects:', err);
      } finally {
        setIsLoadingProjects(false);
      }
    };

    loadProjects();
  }, [isAuthenticated]);

  const handleCreatePresentation = async (title: string, audience: 'executive' | 'expert' | 'investor', template?: File) => {
    try {
      setError(null);
      let templateId: number | null = null;
      
      if (template) {
        const uploadedTemplate = await api.uploadTemplate(template);
        templateId = uploadedTemplate.id;
      }

      const project = await api.createProject({ title, audience, template_id: templateId });
      const newPresentation = projectToPresentation(project);
      setPresentation(newPresentation);
      // Refresh projects list
      const projectList = await api.listProjects();
      const presentations = projectList.map(projectListItemToPresentation);
      setProjects(presentations);
    } catch (err: any) {
      setError(err.message || 'Ошибка создания проекта');
      console.error('Error creating project:', err);
    }
  };

  const handleAddSlide = async () => {
    if (!presentation) return;

    try {
      setError(null);
      const slide = await api.createSlide(presentation.id, {
        title: `Слайд ${presentation.slides.length + 1}`,
        visual_type: 'text',
        prompt: null,
      });

      // Reload full project to get updated slides
      const updatedProject = await api.getProject(presentation.id);
      const updatedPresentation = projectToPresentation(updatedProject);
      setPresentation(updatedPresentation);
      setSelectedSlideId(slide.id);
    } catch (err: any) {
      setError(err.message || 'Ошибка создания слайда');
      console.error('Error creating slide:', err);
    }
  };

  const handleUpdateSlide = async (slideId: number, updates: Partial<Slide>) => {
    if (!presentation) return;

    try {
      setError(null);
      const updateData: any = {};
      if (updates.title !== undefined) {
        updateData.title = updates.title; // Сохраняем как есть, без обрезки
      }
      if (updates.visualType !== undefined) updateData.visual_type = updates.visualType;
      if (updates.prompt !== undefined) updateData.prompt = updates.prompt;

      await api.updateSlide(slideId, updateData);

      // Обновляем локальное состояние сразу, чтобы избежать мерцания
      const updatedSlides = presentation.slides.map(slide => 
        slide.id === slideId ? { ...slide, ...updates } : slide
      );
      setPresentation({ ...presentation, slides: updatedSlides });

      // Затем перезагружаем полный проект для синхронизации
      const updatedProject = await api.getProject(presentation.id);
      const updatedPresentation = projectToPresentation(updatedProject);
      setPresentation(updatedPresentation);
    } catch (err: any) {
      const errorMessage = err?.message || err?.toString() || 'Ошибка обновления слайда';
      setError(typeof errorMessage === 'string' ? errorMessage : 'Ошибка обновления слайда');
      console.error('Error updating slide:', err);
      // Откатываем оптимистичное обновление при ошибке
      const updatedProject = await api.getProject(presentation.id).catch(() => null);
      if (updatedProject) {
        const updatedPresentation = projectToPresentation(updatedProject);
        setPresentation(updatedPresentation);
      }
    }
  };

  const handleDeleteSlide = async (slideId: number) => {
    if (!presentation) return;

    try {
      setError(null);
      await api.deleteSlide(slideId);

      // Reload full project
      const updatedProject = await api.getProject(presentation.id);
      const updatedPresentation = projectToPresentation(updatedProject);
      setPresentation(updatedPresentation);

      if (selectedSlideId === slideId) {
        setSelectedSlideId(updatedPresentation.slides.length > 0 ? updatedPresentation.slides[0].id : null);
      }
    } catch (err: any) {
      setError(err.message || 'Ошибка удаления слайда');
      console.error('Error deleting slide:', err);
    }
  };

  const handleDuplicateSlide = async (slideId: number) => {
    if (!presentation) return;

    try {
      setError(null);
      const slideToDuplicate = presentation.slides.find(s => s.id === slideId);
      if (!slideToDuplicate) return;

      const newSlide = await api.createSlide(presentation.id, {
        title: `${slideToDuplicate.title} (копия)`,
        visual_type: slideToDuplicate.visualType,
        prompt: slideToDuplicate.prompt || null,
      });

      // Reload full project
      const updatedProject = await api.getProject(presentation.id);
      const updatedPresentation = projectToPresentation(updatedProject);
      setPresentation(updatedPresentation);
      setSelectedSlideId(newSlide.id);
    } catch (err: any) {
      setError(err.message || 'Ошибка дублирования слайда');
      console.error('Error duplicating slide:', err);
    }
  };

  const handleGenerateSlide = async (slideId: number) => {
    if (!presentation) return;

    const slide = presentation.slides.find(s => s.id === slideId);
    if (!slide || !slide.prompt.trim()) return;

    // Сразу устанавливаем состояние генерации для немедленной анимации
    setPresentation({
      ...presentation,
      slides: presentation.slides.map(s =>
        s.id === slideId ? { ...s, isGenerating: true } : s
      ),
    });

    try {
      setError(null);
      // Start generation job
      const { job_id } = await api.generateSlide(slideId);
      
      // Poll job status
      const pollJob = async () => {
        const maxAttempts = 60; // 5 minutes max
        let attempts = 0;
        
        const interval = setInterval(async () => {
          attempts++;
          try {
            const job = await api.getJob(job_id);
            
            if (job.status === 'done' || job.status === 'completed') {
              clearInterval(interval);
              // Reload project to get updated content
              const updatedProject = await api.getProject(presentation.id);
              const updatedPresentation = projectToPresentation(updatedProject);
              setPresentation(updatedPresentation);
            } else if (job.status === 'error' || job.status === 'failed') {
              clearInterval(interval);
              setError(job.error_message || 'Ошибка генерации контента');
              // Reload project
              const updatedProject = await api.getProject(presentation.id);
              const updatedPresentation = projectToPresentation(updatedProject);
              setPresentation(updatedPresentation);
            } else if (attempts >= maxAttempts) {
              clearInterval(interval);
              setError('Таймаут генерации контента');
            } else {
              // Update generating status
              const updatedProject = await api.getProject(presentation.id);
              const updatedPresentation = projectToPresentation(updatedProject);
              setPresentation(updatedPresentation);
            }
          } catch (err: any) {
            clearInterval(interval);
            setError(err.message || 'Ошибка проверки статуса генерации');
            console.error('Error polling job:', err);
          }
        }, 5000); // Poll every 5 seconds
      };

      pollJob();
    } catch (err: any) {
      // Сбрасываем состояние генерации при ошибке
      if (presentation) {
        setPresentation({
          ...presentation,
          slides: presentation.slides.map(s =>
            s.id === slideId ? { ...s, isGenerating: false } : s
          ),
        });
      }
      setError(err.message || 'Ошибка запуска генерации');
      console.error('Error generating slide:', err);
    }
  };

  const handleReorderSlides = async (startIndex: number, endIndex: number) => {
    if (!presentation) return;

    try {
      setError(null);
      const result = Array.from(presentation.slides);
      const [removed] = result.splice(startIndex, 1);
      result.splice(endIndex, 0, removed);

      const slideIds = result.map(s => s.id);
      await api.reorderSlides(presentation.id, slideIds);

      // Reload full project
      const updatedProject = await api.getProject(presentation.id);
      const updatedPresentation = projectToPresentation(updatedProject);
      setPresentation(updatedPresentation);
    } catch (err: any) {
      setError(err.message || 'Ошибка изменения порядка слайдов');
      console.error('Error reordering slides:', err);
    }
  };

  const handleExport = async (format: 'pptx' | 'pdf') => {
    if (!presentation) return;

    try {
      setError(null);
      const { job_id } = await api.exportProject(presentation.id, format);
      
      // Poll job status until done
      const pollJob = async () => {
        const maxAttempts = 120; // 10 minutes max
        let attempts = 0;
        
        const interval = setInterval(async () => {
          attempts++;
          try {
            const job = await api.getJob(job_id);
            
            if (job.status === 'done' || job.status === 'completed') {
              clearInterval(interval);
              if (job.result_file_id) {
                // Download file
                const blob = await api.downloadFile(job.result_file_id);
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `${presentation.title}.${format}`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
              }
            } else if (job.status === 'error' || job.status === 'failed') {
              clearInterval(interval);
              setError(job.error_message || 'Ошибка экспорта');
            } else if (attempts >= maxAttempts) {
              clearInterval(interval);
              setError('Таймаут экспорта');
            }
          } catch (err: any) {
            clearInterval(interval);
            setError(err.message || 'Ошибка проверки статуса экспорта');
            console.error('Error polling export job:', err);
          }
        }, 5000); // Poll every 5 seconds
      };

      pollJob();
    } catch (err: any) {
      setError(err.message || 'Ошибка экспорта');
      console.error('Error exporting project:', err);
    }
  };

  const handleReset = () => {
    setPresentation(null);
    setSelectedSlideId(null);
    setShowExportDialog(false);
  };

  const handleOpenProject = async (project: Presentation) => {
    try {
      setError(null);
      const fullProject = await api.getProject(project.id);
      const fullPresentation = projectToPresentation(fullProject);
      setPresentation(fullPresentation);
      setSelectedSlideId(fullPresentation.slides.length > 0 ? fullPresentation.slides[0].id : null);
    } catch (err: any) {
      setError(err.message || 'Ошибка загрузки проекта');
      console.error('Error loading project:', err);
    }
  };

  const handleDeleteProject = async (projectId: number) => {
    try {
      setError(null);
      await api.deleteProject(projectId);
      
      // Refresh projects list
      const projectList = await api.listProjects();
      const presentations = projectList.map(projectListItemToPresentation);
      setProjects(presentations);
      
      // If deleted project was open, close it
      if (presentation?.id === projectId) {
        setPresentation(null);
        setSelectedSlideId(null);
      }
    } catch (err: any) {
      setError(err.message || 'Ошибка удаления проекта');
      console.error('Error deleting project:', err);
    }
  };

  // Show loading screen while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Загрузка...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, show login screen
  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  const allSlidesGenerated = (presentation?.slides.length ?? 0) > 0 && 
    presentation?.slides.every(slide => slide.status === 'completed');

  const selectedSlide = selectedSlideId 
    ? presentation?.slides.find(s => s.id === selectedSlideId) 
    : null;

  if (!presentation) {
    return (
      <>
        {error && (
          <div className="fixed top-4 right-4 bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700 z-50 max-w-md">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-500 hover:text-red-700">×</button>
          </div>
        )}
        <Dashboard 
          projects={projects}
          onCreateProject={handleCreatePresentation}
          onOpenProject={handleOpenProject}
          onDeleteProject={handleDeleteProject}
          currentUser={user?.login || null}
          onLogout={logout}
          isLoading={isLoadingProjects}
        />
      </>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {error && (
        <div className="fixed top-4 right-4 bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700 z-50 max-w-md">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-500 hover:text-red-700">×</button>
        </div>
      )}
      <PresentationHeader
        presentation={presentation}
        onUpdateTitle={async (title) => {
          try {
            await api.updateProject(presentation.id, { title });
            const updatedProject = await api.getProject(presentation.id);
            const updatedPresentation = projectToPresentation(updatedProject);
            setPresentation(updatedPresentation);
          } catch (err: any) {
            setError(err.message || 'Ошибка обновления названия');
          }
        }}
        onExport={() => setShowExportDialog(true)}
        onReset={handleReset}
        canExport={allSlidesGenerated}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Slides List */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <Button
              onClick={handleAddSlide}
              className="w-full gap-2"
            >
              <Plus className="w-4 h-4" />
              Добавить слайд
            </Button>
          </div>

          <SlideList
            slides={presentation.slides}
            selectedSlideId={selectedSlideId}
            onSelectSlide={setSelectedSlideId}
            onDeleteSlide={handleDeleteSlide}
            onDuplicateSlide={handleDuplicateSlide}
            onReorderSlides={handleReorderSlides}
          />
        </div>

        {/* Middle Panel - Slide Editor */}
        <div className="flex-1 overflow-y-auto bg-gray-50">
          {selectedSlide ? (
            <SlideEditor
              slide={selectedSlide}
              onUpdateSlide={async (updates) => {
                // Update slide metadata (title, prompt, visualType)
                if (updates.title !== undefined || updates.prompt !== undefined || updates.visualType !== undefined) {
                  await handleUpdateSlide(selectedSlide.id, updates);
                }
                // Save content separately via API
                if (updates.generatedContent !== undefined && presentation) {
                  try {
                    await api.updateSlideContent(selectedSlide.id, updates.generatedContent);
                    // Reload project to get updated content
                    const updatedProject = await api.getProject(presentation.id);
                    const updatedPresentation = projectToPresentation(updatedProject);
                    setPresentation(updatedPresentation);
                  } catch (err: any) {
                    setError(err.message || 'Ошибка сохранения контента');
                  }
                }
              }}
              onGenerateSlide={() => handleGenerateSlide(selectedSlide.id)}
              onUploadDocument={async (file) => {
                if (!presentation || !selectedSlide) return;
                try {
                  await api.uploadSlideDocument(selectedSlide.id, file);
                  // Reload project to get updated documents
                  const updatedProject = await api.getProject(presentation.id);
                  const updatedPresentation = projectToPresentation(updatedProject);
                  setPresentation(updatedPresentation);
                } catch (err: any) {
                  setError(err.message || 'Ошибка загрузки документа');
                }
              }}
              onDeleteDocument={async (docId) => {
                if (!presentation || !selectedSlide) return;
                try {
                  await api.deleteSlideDocument(selectedSlide.id, docId);
                  // Reload project to get updated documents
                  const updatedProject = await api.getProject(presentation.id);
                  const updatedPresentation = projectToPresentation(updatedProject);
                  setPresentation(updatedPresentation);
                } catch (err: any) {
                  setError(err.message || 'Ошибка удаления документа');
                }
              }}
            />
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Plus className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg mb-2 text-gray-900">Нет выбранного слайда</h3>
                <p className="text-gray-600 mb-4">Добавьте новый слайд или выберите существующий</p>
                <Button onClick={handleAddSlide} className="gap-2">
                  <Plus className="w-4 h-4" />
                  Добавить слайд
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Preview */}
        <div className="w-96 bg-white border-l border-gray-200 overflow-y-auto">
          {selectedSlide ? (
            <SlidePreview slide={selectedSlide} />
          ) : (
            <div className="h-full flex items-center justify-center p-6">
              <p className="text-sm text-gray-500 text-center">
                Выберите слайд для просмотра превью
              </p>
            </div>
          )}
        </div>
      </div>

      {showExportDialog && presentation && (
        <ExportDialog
          open={showExportDialog}
          onOpenChange={setShowExportDialog}
          presentationTitle={presentation.title}
          slideCount={presentation.slides.length}
          onExport={handleExport}
        />
      )}
    </div>
  );
}