import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def extract_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts 40+ advanced audio features for deepfake detection.
    """
    features = {}
    
    # 1. MFCCs (40 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
    features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
    features['mfcc_max'] = np.max(mfccs, axis=1).tolist()
    
    # 2. Mel-Spectrogram (80 bins)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_mean'] = np.mean(mel_spec_db, axis=1).tolist()
    features['mel_std'] = np.std(mel_spec_db, axis=1).tolist()
    
    # 3. Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
    features['spectral_flatness_std'] = float(np.std(spectral_flatness))
    features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1).tolist()
    
    # 4. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # 5. Pitch (Fundamental Frequency) using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) > 0:
        features['pitch_mean'] = float(np.mean(f0_clean))
        features['pitch_std'] = float(np.std(f0_clean))
        features['pitch_min'] = float(np.min(f0_clean))
        features['pitch_max'] = float(np.max(f0_clean))
        features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
        features['pitch_skewness'] = float(skew(f0_clean))
        features['pitch_kurtosis'] = float(kurtosis(f0_clean))
        
        # Pitch stability (low std = AI-like)
        features['pitch_stability'] = float(features['pitch_std'] / (features['pitch_mean'] + 1e-6))
    else:
        features.update({
            'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_min': 0.0,
            'pitch_max': 0.0, 'pitch_range': 0.0, 'pitch_skewness': 0.0,
            'pitch_kurtosis': 0.0, 'pitch_stability': 0.0
        })
    
    # 6. Prosody Features
    # Speaking rate (approximated by zero-crossings and energy)
    rms_energy = librosa.feature.rms(y=y)
    features['energy_mean'] = float(np.mean(rms_energy))
    features['energy_std'] = float(np.std(rms_energy))
    
    # Pauses and silence (energy below threshold)
    silence_threshold = np.percentile(rms_energy, 20)
    silence_frames = np.sum(rms_energy < silence_threshold)
    features['silence_ratio'] = float(silence_frames / len(rms_energy[0]))
    
    # 7. Formants (approximated using LPC)
    # We'll use spectral peaks as a proxy for formants
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    peaks, _ = find_peaks(fft, height=np.max(fft) * 0.1)
    formant_freqs = freqs[peaks][:5] if len(peaks) >= 5 else list(freqs[peaks]) + [0.0] * (5 - len(peaks))
    features['formants'] = formant_freqs.tolist() if hasattr(formant_freqs, 'tolist') else list(formant_freqs)
    
    # 8. Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
    
    # 9. Temporal Features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength_mean'] = float(np.mean(onset_env))
    features['onset_strength_std'] = float(np.std(onset_env))
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    
    # 10. Harmonic-Percussive Separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features['harmonic_ratio'] = float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y)) + 1e-6))
    features['percussive_ratio'] = float(np.sum(np.abs(y_percussive)) / (np.sum(np.abs(y)) + 1e-6))
    
    # 11. Jitter and Shimmer (voice quality)
    # Simplified approximation
    if len(f0_clean) > 1:
        # Jitter: pitch period variability
        jitter = np.std(np.diff(1.0 / (f0_clean + 1e-6)))
        features['jitter'] = float(jitter)
        
        # Shimmer: amplitude variability
        shimmer = np.std(np.diff(rms_energy[0])) / (np.mean(rms_energy) + 1e-6)
        features['shimmer'] = float(shimmer)
    else:
        features['jitter'] = 0.0
        features['shimmer'] = 0.0
    
    # 12. Harmonic-to-Noise Ratio (HNR) approximation
    hnr = features['harmonic_ratio'] / (features['percussive_ratio'] + 1e-6)
    features['hnr'] = float(hnr)
    
    # 13. Statistical moments of spectral features
    features['spectral_centroid_skewness'] = float(skew(spectral_centroids[0]))
    features['spectral_centroid_kurtosis'] = float(kurtosis(spectral_centroids[0]))
    
    return features
