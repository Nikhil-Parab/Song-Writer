# AI Music Generator 🎵

An AI system that creates complete songs based on genre selection.

## Overview

This project generates original songs from scratch. Specify a genre, and the AI will:

- Write appropriate lyrics
- Create instrumental music
- Synthesize vocals
- Mix everything into a complete song

## Features

- Multiple genre support (rap, jazz, rock, pop, R&B, funky, country, electronic)
- Complete song generation (lyrics, instruments, vocals)
- Simple command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/Nikhil-Parab/Song-Writer.git
cd Song-Writer

# Install dependencies
pip install numpy torch transformers librosa soundfile pydub nltk scipy tensorflow TTS
```

## Usage

Run the main script:

```bash
python music_generator.py
```

Follow the prompts to select a genre and create your song.

## How It Works

The system uses:

- **Gemini API** for lyrics generation
- Custom algorithms for music creation
- Text-to-speech technology for vocals
- Audio processing for final mixing
