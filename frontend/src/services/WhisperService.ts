// src/services/WhisperService.ts

// Interfaces for our responses
interface TranscriptionResponse {
  success: boolean;
  transcription: string;
  error?: string;
}

// Check if the backend is available
export const checkWhisperBackendAvailability = async (): Promise<boolean> => {
  try {
    const response = await fetch('http://localhost:5000/health', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (response.ok) {
      const data = await response.json();
      return data.status === 'ok';
    }
    
    return false;
  } catch (error) {
    console.error('Error checking backend availability:', error);
    return false;
  }
};

// Interface for enrollment response
interface EnrollmentResponse {
  success: boolean;
  message?: string;
  error?: string;
}

// Send audio to the Whisper backend for transcription
export const transcribeAudio = async (audioBlob: Blob): Promise<TranscriptionResponse> => {
  const formData = new FormData();
  formData.append('audio', audioBlob);
  
  try {
    const response = await fetch('http://localhost:5000/transcribe', {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error transcribing audio:', error);
    return {
      success: false,
      transcription: '',
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
};

// Send audio to the backend for enrollment
export const enrollPrimaryUser = async (audioBlobs: Blob[]): Promise<EnrollmentResponse> => {
  const formData = new FormData();
  // Append each blob with the same field name 'audio'. Flask's getlist will handle this.
  audioBlobs.forEach((blob, index) => {
      formData.append('audio', blob, `enrollment_clip_${index + 1}.wav`);
  });

  console.log(`Sending ${audioBlobs.length} enrollment files.`); // Debug log

  try {
    const response = await fetch('http://localhost:5000/enroll', {
      method: 'POST',
      body: formData,
      credentials: 'include', // Keep sending credentials for session
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP error ${response.status}`);
    }

    return result;
  } catch (error) {
    console.error('Error enrolling primary user:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown enrollment error occurred'
    };
  }
};