s = input("sequence: ")
octave = int(input("octave: "))

# octave 4
note_table = {
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

for note in s[::2]:
    print(f"note({note_table[note] * 2**(octave-4)}),", end=" ")