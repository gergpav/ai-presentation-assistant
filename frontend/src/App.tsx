import { useState } from 'react';
import { EditorStep } from './components/EditorStep';
import { ExportStep } from './components/ExportStep';
import type { Slide } from './components/SlideEditor';

type Step = 'edit' | 'export';

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('edit');
  const [slides, setSlides] = useState<Slide[]>([]);

  const handleEditorContinue = (editedSlides: Slide[]) => {
    setSlides(editedSlides);
    setCurrentStep('export');
  };

  const handleBackToEditor = () => {
    setCurrentStep('edit');
  };

  return (
    <div className="min-h-screen bg-white">
      {currentStep === 'edit' && (
        <EditorStep onContinue={handleEditorContinue} />
      )}

      {currentStep === 'export' && (
        <ExportStep
          slides={slides}
          onBack={handleBackToEditor}
        />
      )}
    </div>
  );
}
