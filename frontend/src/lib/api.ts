// API Client for backend integration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

// Types matching backend models
export interface User {
  id: number;
  login: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface ProjectListItem {
  id: number;
  title: string;
  audience: 'executive' | 'expert' | 'investor';
  createdAt: string;
  updatedAt: string;
}

export interface SlideDocumentOut {
  id: number;
  name: string;
  type: string;
  size: number;
}

export interface SlideOut {
  id: number;
  title: string;
  prompt: string | null;
  documents: SlideDocumentOut[];
  generatedContent: string | null;
  isGenerating: boolean;
  visualType: 'text' | 'chart' | 'table' | 'image';
  status: 'pending' | 'completed' | 'failed';
}

export interface ProjectOut {
  id: number;
  title: string;
  audience: 'executive' | 'expert' | 'investor';
  slides: SlideOut[];
  createdAt: string;
  updatedAt: string;
  template_id: number | null;
}

export interface ProjectCreate {
  title: string;
  audience: 'executive' | 'expert' | 'investor';
  template_id?: number | null;
}

export interface SlideCreate {
  title?: string;
  visual_type: 'text' | 'chart' | 'table' | 'image';
  prompt?: string | null;
}

export interface SlideUpdate {
  title?: string;
  visual_type?: 'text' | 'chart' | 'table' | 'image';
  prompt?: string | null;
  position?: number;
}

export interface SlideContentUpdate {
  content: string;
}

export interface SlideContentResponse {
  slide_id: number;
  version: number;
  content: string | null;
  created_at: string | null;
}

export interface JobOut {
  id: number;
  status: string;
  progress: number | null;
  result_file_id: number | null;
  error_message: string | null;
}

export interface ExportRequest {
  format: 'pptx' | 'pdf';
}

export interface ExportResponse {
  job_id: number;
}

export interface CreateJobResponse {
  job_id: number;
}

// API Client class
class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    // Load token from localStorage
    this.token = localStorage.getItem('ai-presentations-auth-token');
  }

  setToken(token: string | null) {
    this.token = token;
    if (token) {
      localStorage.setItem('ai-presentations-auth-token', token);
    } else {
      localStorage.removeItem('ai-presentations-auth-token');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const error = await response.json();
        // Обрабатываем разные форматы ошибок
        if (typeof error === 'string') {
          errorMessage = error;
        } else if (error.detail) {
          errorMessage = typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail);
        } else if (error.message) {
          errorMessage = typeof error.message === 'string' ? error.message : JSON.stringify(error.message);
        } else if (error.error) {
          errorMessage = typeof error.error === 'string' ? error.error : JSON.stringify(error.error);
        } else {
          errorMessage = JSON.stringify(error);
        }
      } catch {
        errorMessage = response.statusText || `HTTP error! status: ${response.status}`;
      }
      throw new Error(errorMessage);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return null as T;
    }

    return response.json();
  }

  // Auth endpoints
  async login(login: string, password: string): Promise<TokenResponse> {
    const response = await this.request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ login, password }),
    });
    this.setToken(response.access_token);
    return response;
  }

  async register(login: string, password: string): Promise<User> {
    return this.request<User>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ login, password }),
    });
  }

  async getMe(): Promise<User> {
    return this.request<User>('/auth/me');
  }

  // Projects endpoints
  async listProjects(): Promise<ProjectListItem[]> {
    return this.request<ProjectListItem[]>('/projects');
  }

  async getProject(projectId: number): Promise<ProjectOut> {
    return this.request<ProjectOut>(`/projects/${projectId}`);
  }

  async createProject(data: ProjectCreate): Promise<ProjectOut> {
    return this.request<ProjectOut>('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateProject(projectId: number, data: Partial<ProjectCreate>): Promise<ProjectOut> {
    return this.request<ProjectOut>(`/projects/${projectId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteProject(projectId: number): Promise<void> {
    return this.request<void>(`/projects/${projectId}`, {
      method: 'DELETE',
    });
  }

  // Slides endpoints
  async listSlides(projectId: number): Promise<SlideOut[]> {
    return this.request<SlideOut[]>(`/projects/${projectId}/slides`);
  }

  async createSlide(projectId: number, data: SlideCreate): Promise<SlideOut> {
    return this.request<SlideOut>(`/projects/${projectId}/slides`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateSlide(slideId: number, data: SlideUpdate): Promise<SlideOut> {
    return this.request<SlideOut>(`/slides/${slideId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteSlide(slideId: number): Promise<void> {
    return this.request<void>(`/slides/${slideId}`, {
      method: 'DELETE',
    });
  }

  async reorderSlides(projectId: number, slideIds: number[]): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/projects/${projectId}/slides/reorder`, {
      method: 'POST',
      body: JSON.stringify({ slide_ids: slideIds }),
    });
  }

  // Slide content endpoints
  async getSlideContent(slideId: number): Promise<SlideContentResponse> {
    return this.request<SlideContentResponse>(`/slides/${slideId}/content/latest`);
  }

  async updateSlideContent(slideId: number, content: string): Promise<SlideContentResponse> {
    return this.request<SlideContentResponse>(`/slides/${slideId}/content/latest`, {
      method: 'PUT',
      body: JSON.stringify({ content }),
    });
  }

  // Generate endpoints
  async generateSlide(slideId: number): Promise<CreateJobResponse> {
    return this.request<CreateJobResponse>(`/slides/${slideId}/generate`, {
      method: 'POST',
    });
  }

  // Jobs endpoints
  async getJob(jobId: number): Promise<JobOut> {
    return this.request<JobOut>(`/jobs/${jobId}`);
  }

  // Export endpoints
  async exportProject(projectId: number, format: 'pptx' | 'pdf'): Promise<ExportResponse> {
    return this.request<ExportResponse>(`/projects/${projectId}/export`, {
      method: 'POST',
      body: JSON.stringify({ format }),
    });
  }

  // Documents endpoints
  async listSlideDocuments(slideId: number): Promise<SlideDocumentOut[]> {
    return this.request<SlideDocumentOut[]>(`/slides/${slideId}/documents`);
  }

  async uploadSlideDocument(slideId: number, file: File): Promise<SlideDocumentOut> {
    const formData = new FormData();
    formData.append('upload', file); // Backend expects 'upload' field name

    const url = `${this.baseUrl}/slides/${slideId}/documents`;
    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async deleteSlideDocument(slideId: number, docId: number): Promise<void> {
    return this.request<void>(`/slides/${slideId}/documents/${docId}`, {
      method: 'DELETE',
    });
  }

  // Templates endpoints
  async uploadTemplate(file: File): Promise<{ id: number; filename: string; storage_path: string; created_at: string | null }> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/templates/upload`;
    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Download endpoint
  async downloadFile(fileId: number): Promise<Blob> {
    const url = `${this.baseUrl}/files/${fileId}/download`;
    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(url, {
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.blob();
  }
}

export const api = new ApiClient(API_BASE_URL);
