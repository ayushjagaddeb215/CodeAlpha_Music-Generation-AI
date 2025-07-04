import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import glob

# Function to parse MIDI files and extract note/chord sequences
def get_notes(midi_files):
    notes = []
    for file in midi_files:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = None
            # Select the first instrument (e.g., piano)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            # Extract notes and chords
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    return notes

# Prepare sequences for LSTM
def prepare_sequences(notes, sequence_length=100):
    pitch_names = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitch_names)}
    network_input = []
    network_output = []
    # Create input-output pairs
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    n_vocab = len(pitch_names)
    # Reshape for LSTM [samples, time steps, features]
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)  # Normalize
    network_output = to_categorical(network_output, num_classes=n_vocab)
    return network_input, network_output, pitch_names, note_to_int

# Build LSTM model
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate music sequence
def generate_notes(model, network_input, pitch_names, note_to_int, sequence_length, n_vocab, num_notes=100):
    int_to_note = {number: note for note, number in note_to_int.items()}
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []
    # Generate notes
    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], index / float(n_vocab))
        pattern = pattern.reshape((sequence_length, 1))
    return prediction_output

# Convert generated notes to MIDI file
def create_midi(prediction_output, output_file='output.mid'):
    offset = 0
    output_notes = []
    # Create note and chord objects
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5  # Adjust timing
    # Save to MIDI
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"MIDI file saved as {output_file}")

# Main execution
def main():
    # Specify MIDI files (replace with your dataset path)
    midi_path = "midi_files/*.mid"  # Example: Download classical MIDI from http://www.piano-midi.de/
    midi_files = glob.glob(midi_path)
    if not midi_files:
        print("No MIDI files found. Please download MIDI files and place them in the midi_files directory.")
        return
    
    # Step 1: Extract notes
    print("Parsing MIDI files...")
    notes = get_notes(midi_files)
    if not notes:
        print("No notes extracted from MIDI files.")
        return
    
    # Step 2: Prepare sequences
    sequence_length = 100
    network_input, network_output, pitch_names, note_to_int = prepare_sequences(notes, sequence_length)
    n_vocab = len(pitch_names)
    
    # Step 3: Build and train model
    print("Building and training model...")
    model = create_model(network_input, n_vocab)
    model.fit(network_input, network_output, epochs=20, batch_size=64, verbose=1)
    
    # Step 4: Generate music
    print("Generating music...")
    prediction_output = generate_notes(model, network_input, pitch_names, note_to_int, sequence_length, n_vocab)
    
    # Step 5: Save to MIDI
    print("Saving generated music to MIDI...")
    create_midi(prediction_output, "generated_music.mid")

if __name__ == "__main__":
    main()