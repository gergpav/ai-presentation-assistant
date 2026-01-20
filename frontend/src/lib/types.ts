// Types for frontend components (compatible with backend API types)
import type { ProjectOut, SlideOut, ProjectListItem } from './api';

// Frontend Presentation type (compatible with ProjectOut)
export interface Presentation {
  id: number;
  title: string;
  audience: 'executive' | 'expert' | 'investor';
  slides: Slide[];
  createdAt: Date;
  updatedAt: Date;
  template_id?: number | null;
}

// Frontend Slide type (compatible with SlideOut)
export interface Slide {
  id: number;
  title: string;
  prompt: string;
  documents: SlideDocument[];
  generatedContent: string;
  isGenerating: boolean;
  visualType: 'text' | 'chart' | 'table' | 'image';
  status: 'pending' | 'completed' | 'failed';
}

export interface SlideDocument {
  id: number;
  name: string;
  type: string;
  size: number;
}

// Convert backend ProjectOut to frontend Presentation
export function projectToPresentation(project: ProjectOut): Presentation {
  return {
    id: project.id,
    title: project.title,
    audience: project.audience,
    slides: project.slides.map(slide => ({
      id: slide.id,
      title: slide.title,
      prompt: slide.prompt || '',
      documents: slide.documents.map(doc => ({
        id: doc.id,
        name: doc.name,
        type: doc.type,
        size: doc.size,
      })),
      generatedContent: slide.generatedContent || '',
      isGenerating: slide.isGenerating,
      visualType: slide.visualType,
      status: slide.status,
    })),
    createdAt: new Date(project.createdAt),
    updatedAt: new Date(project.updatedAt),
    template_id: project.template_id,
  };
}

// Convert backend ProjectListItem to frontend Presentation (minimal)
export function projectListItemToPresentation(item: ProjectListItem): Presentation {
  return {
    id: item.id,
    title: item.title,
    audience: item.audience,
    slides: [],
    createdAt: new Date(item.createdAt),
    updatedAt: new Date(item.updatedAt),
  };
}
