import os
import tempfile
import numpy as np
import librosa

def analyze_preview(audio_bytes: bytes) -> dict:
    """
    Extract audio features from a 30-second MP3 preview using librosa.

    Returns:
        energy:  float 0–1  (RMS loudness, normalized)
        tempo:   float       (BPM)
        valence: float 0–1  (brightness + mode proxy for mood)
    """
    # Write to temp file — soundfile can't decode MP3 from BytesIO,
    # but librosa's audioread backend handles MP3 files on disk fine.
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        # Only process 5 seconds of audio to massively speed up the math
        y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=5.0)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Pre-compute STFT once to save massive CPU time instead of computing it 3 times
    S = np.abs(librosa.stft(y))

    # --- Energy: RMS loudness, normalized to 0–1 ---
    rms = librosa.feature.rms(S=S)[0]
    # Typical RMS for music: ~0.02 (quiet acoustic) to ~0.35 (loud EDM/rap)
    raw_energy = float(np.mean(rms))
    energy = min(raw_energy / 0.30, 1.0)  # 0.30 RMS → 1.0 energy

    # --- Tempo: BPM from beat tracking ---
    # SKIPPED: librosa beat tracking is extremely slow and Deezer already provides the BPM!
    tempo = 0.0

    # --- Valence (mood proxy) ---
    # Combine spectral brightness (centroid) with harmonic mode.
    # Brighter + major-key songs tend to sound "happier".

    # Spectral centroid → brightness, normalized 0–1
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    # Typical centroid range: 500–5000 Hz for music
    brightness = float(np.mean(centroid))
    brightness_norm = min(max((brightness - 500) / 4500, 0.0), 1.0)

    # Chroma-based mode estimation (major vs minor)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # Major key indicator: strong 1st, 3rd (major third), 5th intervals
    # Minor key indicator: strong 1st, flat-3rd, 5th intervals
    # Simple heuristic: compare energy at major third vs minor third
    # In chroma, index 4 = major 3rd (E in C scale), index 3 = minor 3rd (Eb)
    root = int(np.argmax(chroma_mean))
    major_third = chroma_mean[(root + 4) % 12]
    minor_third = chroma_mean[(root + 3) % 12]
    mode_score = 0.5  # neutral
    if major_third + minor_third > 0:
        mode_score = float(major_third / (major_third + minor_third))

    # Combine brightness (60% weight) and mode (40% weight)
    valence = brightness_norm * 0.6 + mode_score * 0.4

    return {
        "energy":  round(energy, 4),
        "tempo":   round(tempo, 2),
        "valence": round(valence, 4),
    }
