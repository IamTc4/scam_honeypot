import base64
import io
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Dict

TARGET_SR = 16000

def process_audio(base64_audio: str) -> np.ndarray:
    """
    Decodes base64 audio, converts to WAV, resamples to 16kHz, and trims silence.
    Returns the audio time series as a numpy array.
    """
    try:
        # Decode base64
        audio_data = base64.b64decode(base64_audio)
        
        # Load audio using pydub (handles various formats)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Convert to wav bytes
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Load with librosa for resampling
        y, sr = librosa.load(wav_io, sr=TARGET_SR)
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        return y_trimmed
    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")
