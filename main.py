import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

tau = 2 * np.pi
rate = 40000

instruments = {
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


def f_series(coefficients=[1]):
    return lambda x: sum([c * np.sin(x * (i + 1)) for i, c in enumerate(coefficients)])


def envelope(attack, strength, decay, note_duration, release):
    sustain = note_duration - attack - decay - release

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


def note(
    frequency=0,
    note_duration=0.26,
    attack=0.05,
    strength=0.7,
    decay=0.05,
    release=0.05,
    coefficients=instruments["Clarinet"],
):
    time = np.arange(0, rate * note_duration) / rate

    return [
        envelope(attack, strength, decay, note_duration, release)(t) for t in time
    ] * (f_series(coefficients)(tau * frequency * time))


def note_sequence(
    note_list,
    note_duration=0.26,
    attack=0.05,
    strength=0.7,
    decay=0.05,
    release=0.05,
    coefficients=instruments["Oboe"],
):
    return np.concatenate(
        [
            note(
                frequency,
                duration * note_duration,
                attack,
                strength,
                decay,
                release,
                coefficients,
            )
            for frequency, duration in note_list
        ]
    )


data = np.concatenate(
    [
        note(532),
        note(),
        note(784),
        note(740),
        note(698),
        note(622),
        note(587),
        note(523),
        note(494),
        note(523),
        note(494),
        note(523),
        note(494),
        note(),
        note(),
        note(),
        note(523),
        note(),
        note(784),
        note(740),
        note(698),
        note(622),
        note(587),
        note(523),
        note(523),
        note(587),
        note(622),
        note(698),
        note(740),
        note(),
        note(784),
        note(),
        note(523),
        note(),
        note(784),
        note(740),
        note(698),
        note(622),
        note(587),
        note(523),
        note(494),
        note(523),
        note(494),
        note(523),
        note(494),
    ]
)

data = np.int32(data / data.max() * 2**8) * 2**22
write("test.wav", rate, data)

plt.style.use("dark_background")
plt.axis("off")

plt.plot(data, color="white")
plt.show()
