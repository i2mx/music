from enum import Enum
from dataclasses import dataclass
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


TAU = 2 * np.pi
RATE = 44100

# coefficients for the fourier series of various instruments
instruments = {
    "Average": [0.7, 0.33, 0.23, 0.14, 0.15, 0.105, 0.1, 0.07, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01],
    "Piano": [1, 0.1, 0.31, 0.07, 0.06, 0.05, 0, 0.01, 0.02, 0, 0.01],
    "Guitar": [1, 0.675, 1.26, 0.125, 0.125, 0.12, 0, 0, 0.18, 0.05, 0.01],
    "Horn": [1, 0.675, 1.26, 0.125, 0.125, 0.12, 0, 0, 0.18, 0.05, 0.01],
    "Clarinet": [1, 0.39, 0.205, 0, 0.075, 0.2, 0.075],
    "Oboe": [1, 0.95, 2.05, 0.2, 0.22, 0.23, 0.58, 0.3, 0.22, 0.01],
    "Flute": [1, 9, 3.75, 2.85, 0.3, 0.15, 0.1, 0.12, 0.1],
    "Marimba": [1, 0, 0.6, 0, 0.5, 0, 0.1],  # this one sucks
    "Digital": [1, 1 / 9, 1 / 81],
    "Sine": [1],
    "Triangle": [1, 0, 1 / 9, 0, 1 / 25, 0, 1 / 49, 0, 1 / 81],
    "Square": [
            1,
            0,
            1 / 3,
            0,
            1 / 5,
            0,
            1 / 7,
            0,
            1 / 9,
            0,
            1 / 11,
            0,
            1 / 13,
            0,
            1 / 15,
            0,
            1 / 17,
            0,
            1 / 19,
            0,
            1 / 21,
            0,
            1 / 23,
            0,
            1 / 25,
    ],
    "Saw": [
        1,
        1 / 2,
        1 / 3,
        1 / 4,
        1 / 5,
        1 / 6,
        1 / 7,
        1 / 8,
        1 / 9,
        1 / 10,
        1 / 11,
        1 / 12,
        1 / 13,
        1 / 14,
        1 / 15,
        1 / 16,
        1 / 17,
        1 / 18,
    ],
}

# note table
TB = note_table = {
    "c": 262,
    "C": 277,
    "d": 294,
    "D": 311, 
    "e": 330,
    "f": 349,
    "F": 370,
    "g": 392,
    "G": 415,
    "a": 440,
    "A": 466,
    "b": 494,
    "-": 0,
}

SEMITONE = 2 ** (1 / 12)


@dataclass
class TN:
    at: float
    audio: float



class ChordType(Enum):
    Maj = (1, SEMITONE**4, SEMITONE**7)
    Min = (1, SEMITONE**3, SEMITONE**7)
    M7 = (1, SEMITONE**4, SEMITONE**7, SEMITONE**10)
    Min7 = (1, SEMITONE**3, SEMITONE**7, SEMITONE**10)
    Maj7 = (1, SEMITONE**4, SEMITONE**7, SEMITONE**11)
    Dim = (1, SEMITONE**3, SEMITONE**6)
    Sus4 = (1, SEMITONE**2, SEMITONE**7)
    Sus2 = (1, SEMITONE**5, SEMITONE**7)
    


# returns a function which calculates the combined sin wave
def f_series(coefficients=[1]):
    return lambda x: sum([c * np.sin(x * (i + 1)) for i, c in enumerate(coefficients)])


# return a function which gives the height of the envelope at time t where t = 0 is the start of the note
def envelope(attack, strength, decay, note_duration, release):
    sustain = note_duration - attack - decay

    def f(t):
        if t < attack:
            return t / attack
        if t < attack + decay:
            return 1 + (t - attack) * (strength - 1) / decay
        if t < attack + decay + sustain:
            return strength
        if t < attack + decay + sustain + release:
            return strength - (strength / release) * (t - attack - decay - sustain)
        return 0

    return f


# returns the waveform of a note starting at time 0 (-> audio)
def note(
    frequency=0,
    note_duration=0.26,
    attack=0.05,
    strength=0.7,
    decay=0.05,
    release=0.05,
    coefficients=instruments["Digital"],
):
    time = np.arange(0, RATE * (note_duration + release)) / RATE

    return [
        envelope(attack, strength, decay, note_duration, release)(t) for t in time
    ] * (f_series(coefficients)(TAU * frequency * time))


# a chord is a combination of multiple notes (-> audio)
def chord(base, chord_type,
    delay = 0.3,
    note_duration=1,
    attack=0.05,
    strength=0.7,
    decay=0.05,
    release=0.6,
    coefficients=instruments["Digital"]):
    
    return play_notes([
        TN(
            delay * j , 
            note(base * i, note_duration - delay*j, attack, strength, decay, release, coefficients)) 
        for j, i in enumerate(chord_type.value)
    ])

def arpeggio(base, chord_type, delay=0.3, total_duration=1, note_duration=0.26, attack=0.05, strength=0.7, decay=0.05, release=0.1, coefficients=instruments["Digital"]):
    return play_notes([
        TN(
            delay * i, 
            note(base * chord_type.value[i % len(chord_type.value)], 
                 note_duration, 
                 attack, strength, 
                 decay, release, 
                 coefficients
                 )
        ) 
        for i in range(int(total_duration / delay))
    ])


# takes a list of timed notes and combines it to form a audio signal (TN -> audio)
def play_notes(notes : list[TN]):
    length = int(max([note.at * RATE + len(note.audio) for note in notes]))
    data = np.zeros(length)
    for note in notes:
        data[int(note.at * RATE) : int(note.at * RATE) + len(note.audio)] += note.audio
    return data


# takes a list of audio signals and combines them (audio -> audio)
def combine(tracks):
    length = max([len(track) for track in tracks])
    data = np.zeros(length)
    for track in tracks:
        data[:len(track)] += track
    return data

bg_section = [
    TN(0, 0.5 * chord(TB['g'], ChordType.Maj, delay=0.2, note_duration=1.2,)),
    TN(0, arpeggio(TB['g'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(1.2, 0.5 * chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=1.2)),
    TN(1.2, arpeggio(TB['a'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(2.4, chord(TB['F'], ChordType.Min, delay=0.2, note_duration=1.2)),
    TN(2.4, arpeggio(TB['F'], ChordType.Min, delay=0.2, total_duration=1.2)),
    TN(3.6, chord(TB['b'], ChordType.Maj, delay=0.2, note_duration=0.6)),
    TN(4.2, chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=0.6)),
   
    TN(4.8, 0.5 * chord(TB['g'], ChordType.Maj, delay=0.2, note_duration=1.2,)),
    TN(4.8, arpeggio(TB['g'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(6.0, 0.5 * chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=1.2)),
    TN(6.0, arpeggio(TB['a'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(7.2, chord(TB['F'], ChordType.Min, delay=0.2, note_duration=1.2)),
    TN(7.2, arpeggio(TB['F'], ChordType.Min, delay=0.2, total_duration=1.2)),
    TN(8.4, chord(TB['b'], ChordType.Maj, delay=0.2, note_duration=0.6)),
    TN(9.0, chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=0.6)),

    TN(9.6, 0.5 * chord(TB['g'], ChordType.Maj, delay=0.2, note_duration=1.2,)),
    TN(9.6, arpeggio(TB['g'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(10.8, 0.5 * chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=1.2)),
    TN(10.8, arpeggio(TB['a'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(12.0, chord(TB['F'], ChordType.Min, delay=0.2, note_duration=1.2)),
    TN(12.0, arpeggio(TB['F'], ChordType.Min, delay=0.2, total_duration=1.2)),
    TN(13.2, chord(TB['b'], ChordType.Maj, delay=0.2, note_duration=0.6)),
    TN(13.8, chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=0.6)),

    TN(14.4, chord(TB['g'], ChordType.Maj, delay=0.2, note_duration=1.2)),
    TN(14.4, arpeggio(TB['g'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(15.6, chord(TB['a'], ChordType.Maj, delay=0.2, note_duration=1.2)),
    TN(15.6, arpeggio(TB['a'], ChordType.Maj, delay=0.2, total_duration=1.2)),
    TN(16.8, chord(TB['b'], ChordType.Sus4, delay=0.1, note_duration=1.2)),
    TN(18, chord(TB['b'], ChordType.Maj, delay=0, note_duration=1.2))
]

bg_data = bg_section + [TN(tn.at + 19.2, tn.audio) for tn in bg_section] + [TN(tn.at + 2 * 19.2, tn.audio) for tn in bg_section] + [TN(tn.at + 3 * 19.2, tn.audio) for tn in bg_section]
bg_audio = play_notes(bg_data)

fg_audio = play_notes([
    TN()
])

data = combine([bg_audio])

data = np.int32(data / data.max() * 2**8) * 2**22
write("test.wav", RATE, data)

plt.style.use("dark_background")
plt.axis("off")

plt.plot(data, color="white")
plt.show()
