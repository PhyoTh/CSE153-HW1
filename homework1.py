import numpy as np
from scipy.io import wavfile
import glob
from mido import MidiFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math

SAMPLE_RATE = 44100

# Part A - Sine Wave Generation

# Q1
def note_name_to_frequency(note_name):
    note_map = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    # Parse note name: could be like "C#4" or "A4"
    if len(note_name) > 1 and note_name[1] == '#':
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])

    semitone = note_map[note]
    # A4 = 440 Hz, A is semitone 9 in octave 4
    half_steps_from_a4 = (semitone - 9) + (octave - 4) * 12
    frequency = 440.0 * (2.0 ** (half_steps_from_a4 / 12.0))
    return frequency


# Q2
def decrease_amplitude(audio):
    envelope = np.linspace(1, 0, len(audio))
    return audio * envelope


# Q3
def add_delay_effects(audio):
    delay_samples = int(0.5 * SAMPLE_RATE)
    total_length = len(audio) + delay_samples
    delayed_audio = np.zeros(total_length)
    delayed_audio[:len(audio)] += 0.7 * audio
    delayed_audio[delay_samples:delay_samples + len(audio)] += 0.3 * audio
    return delayed_audio


# Q4
def concatenate_audio(list_of_your_audio):
    return np.concatenate(list_of_your_audio)


def mix_audio(list_of_your_audio, amplitudes):
    result = np.zeros_like(list_of_your_audio[0], dtype=float)
    for audio, amp in zip(list_of_your_audio, amplitudes):
        result = result + amp * audio
    return result


# Q5
def create_sawtooth_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)
    for k in range(1, 20):  # k=1 to 19
        wave += ((-1) ** (k + 1)) / k * np.sin(2 * np.pi * k * frequency * t)
    wave *= 2 / np.pi
    return wave


# Part B - Binary Classification

def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files


# Q6
def get_num_beats(file_path):
    mid = MidiFile(file_path)
    ticks_per_beat = mid.ticks_per_beat
    max_ticks = 0
    for track in mid.tracks:
        cumulative_ticks = 0
        for msg in track:
            cumulative_ticks += msg.time
        if cumulative_ticks > max_ticks:
            max_ticks = cumulative_ticks
    nBeats = max_ticks / ticks_per_beat
    return nBeats


def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = []
    drum_beat_nums = []
    for file_path in piano_path_list:
        piano_beat_nums.append(get_num_beats(file_path))
    for file_path in drum_path_list:
        drum_beat_nums.append(get_num_beats(file_path))

    return {
        "piano_midi_num": len(piano_path_list),
        "drum_midi_num": len(drum_path_list),
        "average_piano_beat_num": np.average(piano_beat_nums) if piano_beat_nums else 0,
        "average_drum_beat_num": np.average(drum_beat_nums) if drum_beat_nums else 0,
    }


# Q7
def get_lowest_pitch(file_path):
    mid = MidiFile(file_path)
    lowest = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if lowest is None or msg.note < lowest:
                    lowest = msg.note
    return lowest


def get_highest_pitch(file_path):
    mid = MidiFile(file_path)
    highest = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if highest is None or msg.note > highest:
                    highest = msg.note
    return highest


def get_unique_pitch_num(file_path):
    mid = MidiFile(file_path)
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.add(msg.note)
    return len(pitches)


# Q8
def get_average_pitch_value(file_path):
    mid = MidiFile(file_path)
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    if len(notes) == 0:
        return 0.0
    return np.mean(notes)


# Q9
def featureQ9(file_path):
    return [
        get_lowest_pitch(file_path),
        get_highest_pitch(file_path),
        get_unique_pitch_num(file_path),
        get_average_pitch_value(file_path),
    ]


# Q10
# Additional features beyond Q9: pitch range, pitch std dev, average velocity,
# and velocity std dev. All features are on similar scales (0-127 range) so
# logistic regression converges without feature scaling.
# Piano tends to have wider pitch range and more pitch variation, while drums
# have concentrated pitches and distinct velocity patterns.
def featureQ10(file_path):
    mid = MidiFile(file_path)
    notes = []
    velocities = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)

    lowest = min(notes) if notes else 0
    highest = max(notes) if notes else 0
    unique_count = len(set(notes))
    avg_pitch = np.mean(notes) if notes else 0.0
    pitch_range = highest - lowest
    pitch_std = np.std(notes) if notes else 0.0
    avg_velocity = np.mean(velocities) if velocities else 0.0
    velocity_std = np.std(velocities) if velocities else 0.0

    return [
        lowest,
        highest,
        unique_count,
        avg_pitch,
        pitch_range,
        pitch_std,
        avg_velocity,
        velocity_std,
    ]
