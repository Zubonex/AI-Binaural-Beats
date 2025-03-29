# Spatial Rain with Binaural Beat Generation

This project generates a 10-minute spatialized rain sound combined with a 10Hz binaural beat using AudioLDM, Librosa, and NumPy. The final audio is saved as `spatial_rain_with_binaural_beat.wav`.

## Requirements
Ensure you have the following Python packages installed:

```bash
pip install torch diffusers soundfile librosa numpy
```

## Steps

### 1. Generate Rain Sound with AudioLDM
- Uses the `cvssp/audioldm-s-full-v2` model to generate a short clip of rain sounds.
- Saves the generated audio as `short_rain.wav`.

### 2. Loop Rain Audio for 10 Minutes
- Loads the short rain audio.
- Loops the audio to create a seamless 10-minute rain sound.

### 3. Generate 10Hz Binaural Beat
- Creates a sine wave with 10Hz frequency difference between the left and right channels.
- Applies a fade-in and fade-out effect.
- Introduces spatial panning for an immersive effect.

### 4. Spatialize Rain and Combine with Binaural Beat
- Adds movement to the rain sound through panning.
- Balances rain and binaural beats evenly.

### 5. Save and Play Combined Audio
- Saves the final audio as `spatial_rain_with_binaural_beat.wav`.

## Usage
Run the Python script to generate the audio:

```bash
python source.py
```

The generated file can be played using any media player that supports WAV format.

## Output
- **File:** `spatial_rain_with_binaural_beat.wav`
- **Duration:** 10 minutes
- **Effect:** Relaxing rain sound with a 10Hz binaural beat for relaxation or focus.

Enjoy your immersive audio experience!

