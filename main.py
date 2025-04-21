import os
import numpy as np
import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
import nltk
from nltk.tokenize import word_tokenize
import scipy
import tensorflow as tf
from scipy.io import wavfile
from TTS.api import TTS
import time
import google.generativeai as genai
from dotenv import load_dotenv
# Download necessary NLTK data
nltk.download('punkt', quiet=True)

print("Starting AI Music Generator with Gemini API...")

class LyricsGenerator:
    def __init__(self, api_key):
        print("Initializing Gemini-powered lyrics generator...")
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Genre-specific prompts to guide generation
        self.genre_prompts = {
            "rap": "In the streets of the city where dreams come alive",
            "jazz": "Midnight in the city, stars shine so bright",
            "rock": "Screaming at the sky with my fists up high",
            "pop": "Dancing in the moonlight, feeling so right",
            "rnb": "Baby when we're together, time stands still",
            "funky": "Get on up and move your body to the rhythm",
            "country": "Down the old dirt road, where memories grow",
            "electronic": "Pulses of light in the darkness of night"
        }
    
    def generate_lyrics(self, genre, verse_count=3, chorus_count=1):
        print(f"Generating {genre} lyrics with Gemini...")
        if genre.lower() not in self.genre_prompts:
            genre = "pop"  # Default to pop if genre not found
            
        prompt = self.genre_prompts[genre.lower()]
        
        # Create structured prompt for Gemini
        generation_prompt = f"""
        Write original song lyrics in {genre} style.
        Starting inspiration: {prompt}
        
        Structure:
        - Include {verse_count} verses
        - Include a chorus
        - Format with "Verse X:" before each verse and "Chorus:" before the chorus
        - Keep verses around 4-6 lines each
        - Keep chorus around 4-5 lines
        - Make it sound authentic to the {genre} genre
        - Focus on themes typical for {genre} music
        - Be creative and original
        
        Return ONLY the formatted lyrics.
        """
        
        try:
            # Call Gemini API
            response = self.model.generate_content(generation_prompt)
            lyrics = response.text.strip()
            
            # Ensure we have the proper structure
            if "Verse 1:" not in lyrics:
                lyrics = self._format_lyrics(lyrics, verse_count)
                
            # Add repeated chorus as needed
            for i in range(chorus_count):
                if "Chorus:" in lyrics:
                    chorus_text = lyrics.split("Chorus:")[1].split("\n\n")[0]
                    lyrics += f"\n\nChorus:{chorus_text}"
            
            return lyrics
        
        except Exception as e:
            print(f"Error using Gemini API: {e}")
            print("Falling back to basic lyrics generation")
            return self._fallback_generation(genre, verse_count, chorus_count)
    
    def _format_lyrics(self, raw_lyrics, verse_count):
        """Format lyrics if Gemini didn't return the proper structure"""
        lines = raw_lyrics.split('\n')
        formatted = ""
        
        # Basic formatting with verse/chorus structure
        chunk_size = min(4, len(lines) // (verse_count + 1))
        
        for i in range(verse_count):
            formatted += f"\nVerse {i+1}:\n"
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx <= len(lines):
                formatted += '\n'.join(lines[start_idx:end_idx]) + '\n'
        
        # Add chorus
        formatted += "\nChorus:\n"
        chorus_start = verse_count * chunk_size
        if chorus_start < len(lines):
            formatted += '\n'.join(lines[chorus_start:chorus_start+chunk_size]) + '\n'
        
        return formatted
    
    def _fallback_generation(self, genre, verse_count, chorus_count):
        """Simple fallback if Gemini API fails"""
        prompt = self.genre_prompts[genre.lower()]
        lyrics = f"\nVerse 1:\n{prompt}\nAnd the music plays on\nThrough the night until dawn\nLike a story untold\nIn our hearts we hold\n"
        
        for i in range(1, verse_count):
            lyrics += f"\nVerse {i+1}:\nAnother verse for this {genre} song\nThe rhythm keeps us moving along\nWords flow with the beat\nMaking this track complete\n"
            
        lyrics += "\nChorus:\nThis is where the chorus would be\nCatchy and full of energy\nThe part everyone sings along\nThe heart of our song\n"
        
        # Add additional chorus instances
        for _ in range(chorus_count-1):
            lyrics += "\nChorus:\nThis is where the chorus would be\nCatchy and full of energy\nThe part everyone sings along\nThe heart of our song\n"
            
        return lyrics

class MusicGenerator:
    def __init__(self):
        print("Initializing music generator...")
        # Define basic patterns for different genres
        self.sample_rate = 44100
        self.bpm = 120
        
        # Genre-specific configurations
        self.genre_configs = {
            "rap": {"bpm": 95, "bass_freq": 80, "drum_pattern": "kick-hihat-snare-hihat"},
            "jazz": {"bpm": 120, "bass_freq": 65, "chord_complexity": "high"},
            "rock": {"bpm": 128, "distortion": 0.8, "drum_pattern": "kick-kick-snare-kick-hihat"},
            "pop": {"bpm": 115, "bass_freq": 70, "chord_complexity": "medium"},
            "rnb": {"bpm": 90, "bass_freq": 60, "chord_complexity": "medium-high"},
            "funky": {"bpm": 110, "bass_freq": 75, "drum_pattern": "kick-hihat-snare-hihat-kick"},
            "country": {"bpm": 100, "chord_complexity": "low"},
            "electronic": {"bpm": 140, "bass_freq": 85, "drum_pattern": "kick-hihat-kick-hihat-snare"}
        }
        
    def generate_music(self, genre, duration=30):
        print(f"Generating {genre} instrumental...")
        if genre.lower() not in self.genre_configs:
            genre = "pop"  # Default
            
        config = self.genre_configs[genre.lower()]
        bpm = config.get("bpm", 120)
        
        # Total number of samples for the requested duration
        total_samples = int(self.sample_rate * duration)
        
        # Generate base track components
        beat_track = self._generate_beat_track(genre, total_samples)
        bass_track = self._generate_bass_track(genre, total_samples)
        melody_track = self._generate_melody_track(genre, total_samples)
        
        # Mix the components
        mixed_track = beat_track * 0.4 + bass_track * 0.3 + melody_track * 0.3
        
        # Normalize
        mixed_track = mixed_track / np.max(np.abs(mixed_track))
        
        return mixed_track, self.sample_rate
        
    def _generate_beat_track(self, genre, total_samples):
        config = self.genre_configs[genre.lower()]
        bpm = config.get("bpm", 120)
        
        # Calculate beat length
        beat_length = int(60 / bpm * self.sample_rate)
        
        # Create empty track
        beat_track = np.zeros(total_samples)
        
        # Add kick drum
        kick = self._generate_kick()
        
        # Place kicks according to genre pattern
        if genre.lower() in ["rap", "hiphop"]:
            pattern = [0, 4, 8, 12]
        elif genre.lower() in ["rock", "metal"]:
            pattern = [0, 3, 6, 9, 12]
        elif genre.lower() in ["jazz", "blues"]:
            pattern = [0, 6]
        elif genre.lower() in ["electronic", "techno", "house"]:
            pattern = [0, 2, 4, 6, 8, 10, 12, 14]
        else:  # Default pattern for other genres
            pattern = [0, 4, 8, 12]
            
        # Place kicks in the track
        for i in range(0, total_samples, beat_length * 4):  # One pattern every 4 beats
            for p in pattern:
                pos = i + p * beat_length // 4
                if pos + len(kick) < total_samples:
                    beat_track[pos:pos+len(kick)] += kick
                    
        # Add snare
        snare = self._generate_snare()
        
        # Default snare pattern (typically on beats 2 and 4)
        snare_pattern = [4, 12]
        
        # Place snares in the track
        for i in range(0, total_samples, beat_length * 4):
            for p in snare_pattern:
                pos = i + p * beat_length // 4
                if pos + len(snare) < total_samples:
                    beat_track[pos:pos+len(snare)] += snare
                    
        # Add hi-hat
        hihat = self._generate_hihat()
        
        # Hi-hat pattern depends on genre
        if genre.lower() in ["rap", "hiphop", "trap"]:
            hihat_pattern = [0, 2, 4, 6, 8, 10, 12, 14]  # 16th notes
        elif genre.lower() in ["jazz"]:
            hihat_pattern = [0, 2, 4, 6, 8, 10, 12, 14]  # Swing feel
        else:
            hihat_pattern = [0, 2, 4, 6, 8, 10, 12, 14]  # 8th notes
            
        # Place hi-hats in the track
        for i in range(0, total_samples, beat_length * 4):
            for p in hihat_pattern:
                pos = i + p * beat_length // 4
                if pos + len(hihat) < total_samples:
                    beat_track[pos:pos+len(hihat)] += hihat * 0.7  # Lower volume for hi-hats
                    
        return beat_track
        
    def _generate_kick(self):
        t = np.linspace(0, 0.3, int(0.3 * self.sample_rate))
        frequency = 150 * np.exp(-10 * t)
        amplitude = np.exp(-20 * t)
        kick = amplitude * np.sin(2 * np.pi * frequency * t)
        return kick
        
    def _generate_snare(self):
        t = np.linspace(0, 0.2, int(0.2 * self.sample_rate))
        white_noise = np.random.uniform(-1, 1, len(t))
        envelope = np.exp(-13 * t)
        snare = envelope * white_noise
        return snare
        
    def _generate_hihat(self):
        t = np.linspace(0, 0.1, int(0.1 * self.sample_rate))
        white_noise = np.random.uniform(-1, 1, len(t))
        envelope = np.exp(-50 * t)
        hihat = envelope * white_noise
        return hihat
        
    def _generate_bass_track(self, genre, total_samples):
        config = self.genre_configs[genre.lower()]
        bpm = config.get("bpm", 120)
        base_freq = config.get("bass_freq", 60)
        
        # Calculate beat length
        beat_length = int(60 / bpm * self.sample_rate)
        
        # Create empty track
        bass_track = np.zeros(total_samples)
        
        # Define bass pattern based on genre
        if genre.lower() in ["rap", "hiphop", "trap"]:
            note_pattern = [0, 0, 7, 0, 5, 0, 7, 0]
            note_lengths = [2, 2, 2, 2, 2, 2, 2, 2]  # In 8th notes
        elif genre.lower() in ["funk", "rnb"]:
            note_pattern = [0, 3, 5, 7, 5, 3, 2, 0]
            note_lengths = [1, 1, 1, 1, 1, 1, 1, 1]  # Sixteenth notes for funk
        elif genre.lower() in ["rock"]:
            note_pattern = [0, 0, 7, 7, 5, 5, 3, 3]
            note_lengths = [2, 2, 2, 2, 2, 2, 2, 2]
        else:
            note_pattern = [0, 3, 7, 3, 0, 3, 7, 10]
            note_lengths = [2, 2, 2, 2, 2, 2, 2, 2]
            
        # Generate bass line
        pattern_length = sum(note_lengths) * beat_length // 8
        
        for i in range(0, total_samples, pattern_length):
            pos = i
            for note, length in zip(note_pattern, note_lengths):
                note_length = length * beat_length // 8
                
                if pos + note_length <= total_samples:
                    note_freq = base_freq * 2**(note/12)  # Convert semitones to frequency
                    t = np.linspace(0, note_length/self.sample_rate, note_length)
                    note_wave = 0.5 * np.sin(2 * np.pi * note_freq * t)
                    
                    # Add envelope
                    envelope = np.ones_like(t)
                    attack = int(0.01 * self.sample_rate)
                    release = int(0.05 * self.sample_rate)
                    
                    if len(envelope) > attack:
                        envelope[:attack] = np.linspace(0, 1, attack)
                    if len(envelope) > release:
                        envelope[-release:] = np.linspace(1, 0, release)
                        
                    bass_track[pos:pos+note_length] += note_wave * envelope
                    
                pos += note_length
                
        return bass_track
        
    def _generate_melody_track(self, genre, total_samples):
        config = self.genre_configs[genre.lower()]
        bpm = config.get("bpm", 120)
        chord_complexity = config.get("chord_complexity", "medium")
        
        # Calculate beat length
        beat_length = int(60 / bpm * self.sample_rate)
        
        # Create empty track
        melody_track = np.zeros(total_samples)
        
        # Define chord progression based on genre
        if genre.lower() in ["pop", "rock"]:
            # Simple I-V-vi-IV progression 
            chords = [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]]
        elif genre.lower() in ["jazz"]:
            # More complex jazz progression
            chords = [[0, 4, 7, 11], [5, 9, 12, 16], [7, 11, 14, 17], [0, 3, 7, 10]]
        elif genre.lower() in ["rap", "hiphop", "trap"]:
            # Minor progression common in hip-hop
            chords = [[0, 3, 7], [5, 8, 12], [7, 10, 14], [3, 7, 10]]
        else:
            # Default progression
            chords = [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]]
            
        # Base frequency for melody
        base_freq = 220  # A3
        
        # Generate chord progression
        chord_length = beat_length * 4  # One chord per bar
        
        for i in range(0, total_samples, len(chords) * chord_length):
            for chord_idx, chord in enumerate(chords):
                pos = i + chord_idx * chord_length
                if pos + chord_length > total_samples:
                    break
                    
                # Generate each note in the chord
                for note in chord:
                    note_freq = base_freq * 2**(note/12)
                    t = np.linspace(0, chord_length/self.sample_rate, chord_length)
                    
                    # Create a simple synth sound with harmonics
                    note_wave = 0.2 * np.sin(2 * np.pi * note_freq * t)  # Fundamental
                    note_wave += 0.1 * np.sin(2 * np.pi * note_freq * 2 * t)  # First harmonic
                    note_wave += 0.05 * np.sin(2 * np.pi * note_freq * 3 * t)  # Second harmonic
                    
                    # Add envelope
                    envelope = np.ones_like(t)
                    attack = int(0.05 * self.sample_rate)
                    decay = int(0.1 * self.sample_rate)
                    release = int(0.3 * self.sample_rate)
                    
                    if len(envelope) > attack:
                        envelope[:attack] = np.linspace(0, 1, attack)
                    if len(envelope) > attack + decay:
                        envelope[attack:attack+decay] = np.linspace(1, 0.8, decay)
                    if len(envelope) > release:
                        envelope[-release:] = np.linspace(0.8, 0, release)
                        
                    melody_track[pos:pos+chord_length] += note_wave * envelope
                    
        return melody_track

class VocalSynthesizer:
    def __init__(self):
        print("Initializing vocal synthesizer...")
        try:
            # Initialize TTS system
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            self.ready = True
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            print("Will use basic speech synthesis instead")
            self.ready = False
    
    def generate_vocals(self, lyrics, genre, sample_rate=44100):
        print("Generating vocals...")
        
        if not lyrics:
            return np.zeros(sample_rate * 5)  # Return silence if no lyrics
            
        # Clean up lyrics for TTS
        lyrics_lines = lyrics.split('\n')
        cleaned_lyrics = []
        
        for line in lyrics_lines:
            if line and not line.startswith("Verse") and not line.startswith("Chorus"):
                cleaned_lyrics.append(line)
        
        combined_lyrics = " ".join(cleaned_lyrics)
        
        # Use TTS if available
        if self.ready:
            try:
                # Create temporary file
                temp_file = "temp_vocal.wav"
                
                # Generate speech
                self.tts.tts_to_file(text=combined_lyrics, file_path=temp_file)
                
                # Load the generated audio
                vocal_track, sr = librosa.load(temp_file, sr=sample_rate)
                
                # Remove temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                return vocal_track
            except Exception as e:
                print(f"Error in TTS generation: {e}")
                print("Falling back to basic synthesis")
                
        # If TTS fails or isn't available, use basic synthesis
        return self._basic_synthesis(combined_lyrics, sample_rate)
    
    def _basic_synthesis(self, text, sample_rate):
        """Very basic speech synthesis as fallback"""
        words = word_tokenize(text)
        full_audio = np.array([])
        
        for word in words:
            # Generate a simple tone for each word - not actual speech
            duration = len(word) * 0.15  # Duration based on word length
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Generate a unique frequency based on the word
            freq = 300 + sum(ord(c) for c in word) % 200
            word_audio = 0.3 * np.sin(2 * np.pi * freq * t)
            
            # Add envelope
            envelope = np.ones_like(word_audio)
            attack = int(0.01 * sample_rate)
            release = int(0.05 * sample_rate)
            
            if len(envelope) > attack:
                envelope[:attack] = np.linspace(0, 1, attack)
            if len(envelope) > release:
                envelope[-release:] = np.linspace(1, 0, release)
                
            word_audio = word_audio * envelope
            
            # Add to full audio
            full_audio = np.concatenate([full_audio, word_audio, np.zeros(int(0.1 * sample_rate))])
            
        return full_audio

class AudioMixer:
    def __init__(self):
        print("Initializing audio mixer...")
    
    def mix_track(self, instrumental, vocals, sample_rate=44100):
        print("Mixing final track...")
        
        # Make sure both tracks are the same length
        max_length = max(len(instrumental), len(vocals))
        
        # Pad shorter track
        if len(instrumental) < max_length:
            instrumental = np.pad(instrumental, (0, max_length - len(instrumental)))
        if len(vocals) < max_length:
            vocals = np.pad(vocals, (0, max_length - len(vocals)))
            
        # Simple mixing with genre-appropriate levels
        mixed = instrumental * 0.7 + vocals * 0.5
        
        # Apply some basic mastering
        mixed = self._apply_compression(mixed)
        mixed = self._apply_eq(mixed, sample_rate)
        
        # Normalize to prevent clipping
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
        
        return mixed, sample_rate
    
    def _apply_compression(self, audio, threshold=0.5, ratio=4.0):
        """Apply basic compression"""
        result = np.copy(audio)
        
        # Find samples above threshold
        mask = np.abs(result) > threshold
        
        # Apply compression to those samples
        result[mask] = np.sign(result[mask]) * (
            threshold + (np.abs(result[mask]) - threshold) / ratio
        )
        
        return result
    
    def _apply_eq(self, audio, sample_rate):
        """Apply very basic EQ using FFT"""
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio)
        
        # Get frequency bins
        freq_bins = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        # Apply very simple EQ
        # Boost low frequencies slightly (bass)
        bass_mask = freq_bins < 200
        fft_data[bass_mask] *= 1.2
        
        # Cut very low frequencies (rumble)
        subsonic_mask = freq_bins < 30
        fft_data[subsonic_mask] *= 0.5
        
        # Boost presence (vocals)
        presence_mask = (freq_bins > 2000) & (freq_bins < 5000)
        fft_data[presence_mask] *= 1.1
        
        # Convert back to time domain
        eq_audio = np.fft.irfft(fft_data, len(audio))
        
        return eq_audio

class MusicGenerationSystem:
    def __init__(self, api_key):
        self.lyrics_generator = LyricsGenerator(api_key)
        self.music_generator = MusicGenerator()
        self.vocal_synthesizer = VocalSynthesizer()
        self.audio_mixer = AudioMixer()
        
    def generate_song(self, genre):
        print(f"\nGenerating {genre} song...")
        
        # 1. Generate lyrics with Gemini
        lyrics = self.lyrics_generator.generate_lyrics(genre)
        print("\nGenerated Lyrics:")
        print(lyrics)
        
        # 2. Generate instrumental track
        instrumental, sample_rate = self.music_generator.generate_music(genre, duration=60)
        print(f"\nGenerated {genre} instrumental track (60 seconds)")
        
        # 3. Generate vocals
        vocals = self.vocal_synthesizer.generate_vocals(lyrics, genre, sample_rate)
        print("\nGenerated vocals")
        
        # 4. Mix everything together
        final_mix, sample_rate = self.audio_mixer.mix_track(instrumental, vocals, sample_rate)
        print("\nMixed final track")
        
        # 5. Save the result
        output_filename = f"{genre.lower()}_song_{int(time.time())}.wav"
        sf.write(output_filename, final_mix, sample_rate)
        print(f"\nSaved song to {output_filename}")
        
        return {
            "lyrics": lyrics,
            "filename": output_filename,
            "duration": len(final_mix) / sample_rate
        }

def main():
    print("Welcome to the Gemini-powered AI Music Generator!")
    print("This system creates original songs with AI-generated lyrics, instrumentals, and vocals")
    print("\nAvailable genres: rap, jazz, rock, pop, rnb, funky, country, electronic")
    
    # Get Gemini API key
    load_dotenv()
    api_key =  os.getenv("api_key")
        
    while True:
        genre = input("\nWhat genre of song would you like to create? ")
        
        # Initialize the system with API key
        system = MusicGenerationSystem(api_key)
        
        # Generate the song
        result = system.generate_song(genre)
        
        print("\n====== Song Generation Complete ======")
        print(f"Created a {genre} song!")
        print(f"Duration: {result['duration']:.1f} seconds")
        print(f"Saved to: {result['filename']}")
        
        another = input("\nWould you like to create another song? (y/n): ")
        if another.lower() != 'y':
            break
    
    print("Thank you for using the Gemini-powered AI Music Generator!")

if __name__ == "__main__":
    main()