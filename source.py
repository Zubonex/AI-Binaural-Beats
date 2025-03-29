import torch
from diffusers import AudioLDMPipeline
import soundfile as sf
from IPython.display import Audio
import librosa
import numpy as np

# 1. Generate short rain audio with AudioLDM
# ------------------------------------------
model_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "Gentle rain drops falling on a window"
short_rain_audio = pipe(prompt).audios[0]
short_rain_filename = "short_rain.wav"
sf.write(short_rain_filename, short_rain_audio, samplerate=16000)

# 2. Load and Loop rain audio with librosa
# ------------------------------------------
rain_audio, sr = librosa.load(short_rain_filename)
desired_length_seconds = 600
desired_length_samples = desired_length_seconds * sr
short_audio_length = len(rain_audio)
looped_rain_audio = np.zeros(desired_length_samples, dtype=rain_audio.dtype)
for i in range(desired_length_samples):
    looped_rain_audio[i] = rain_audio[i % short_audio_length]

# 3. Generate Spatialized 10Hz Binaural Beat
# ---------------------------------------------
beat_frequency = 10
carrier_frequency = 200
duration_seconds = 600
fade_in_seconds = 5
fade_out_seconds = 5
t = np.linspace(0, duration_seconds, desired_length_samples)
left_channel = np.sin(2 * np.pi * (carrier_frequency - beat_frequency / 2) * t)
right_channel = np.sin(2 * np.pi * (carrier_frequency + beat_frequency / 2) * t)
binaural_beat = np.column_stack((left_channel, right_channel))

# Apply fade-in and fade-out
fade_in_samples = int(fade_in_seconds * sr)
fade_out_samples = int(fade_out_seconds * sr)
if fade_in_samples > 0:
    fade_in_ramp = np.linspace(0, 1, fade_in_samples)
    binaural_beat[:fade_in_samples, 0] *= fade_in_ramp
    binaural_beat[:fade_in_samples, 1] *= fade_in_ramp
if fade_out_samples > 0:
    fade_out_ramp = np.linspace(1, 0, fade_out_samples)
    binaural_beat[-fade_out_samples:, 0] *= fade_out_ramp
    binaural_beat[-fade_out_samples:, 1] *= fade_out_ramp

# Spatialization parameters (simplified)
pan_speed = 0.1  # Cycles per second (adjust for speed)
pan_width = 0.7  # Maximum panning amount (0 to 1)

# Create panning modulation
pan_modulation = np.sin(2 * np.pi * pan_speed * t) * pan_width

# Apply panning to binaural beat
left_gain = 1 - pan_modulation
right_gain = 1 + pan_modulation
left_gain = np.clip(left_gain, 0, 1) # Ensure gains are within 0 and 1
right_gain = np.clip(right_gain, 0, 1)
binaural_beat[:, 0] *= left_gain
binaural_beat[:, 1] *= right_gain

# 4. Spatialize Rain and Combine with Binaural Beat
# -------------------------------------------------
rain_pan_speed = 0.05  # Slower panning for rain
rain_pan_modulation = np.sin(2 * np.pi * rain_pan_speed * t) * 0.5  # Less extreme panning
rain_left_gain = 1 - rain_pan_modulation
rain_right_gain = 1 + rain_pan_modulation
rain_left_gain = np.clip(rain_left_gain, 0, 1)
rain_right_gain = np.clip(rain_right_gain, 0, 1)

spatialized_rain_left = looped_rain_audio * rain_left_gain
spatialized_rain_right = looped_rain_audio * rain_right_gain
spatialized_rain = np.column_stack((spatialized_rain_left, spatialized_rain_right))

if len(spatialized_rain) > len(binaural_beat):
    spatialized_rain = spatialized_rain[:len(binaural_beat)]
elif len(binaural_beat) > len(spatialized_rain):
    binaural_beat = binaural_beat[:len(spatialized_rain)]

mixed_audio = (spatialized_rain * 0.5 + binaural_beat * 0.5)
mixed_audio = mixed_audio.astype(np.float32)

# 5. Save and Play Combined Audio
# --------------------------------
combined_filename = "spatial_rain_with_binaural_beat.wav"
sf.write(combined_filename, mixed_audio, sr, format='WAV')
print(f"Combined audio saved: {combined_filename}")
Audio(mixed_audio.T, rate=sr)
