import mido
from itertools import groupby

CHORD_PATTERNS = {
    (0, 4, 7): 'major',
    (0, 3, 7): 'minor',
    (0, 3, 6): 'dim',
    (0, 4, 8): 'aug',
    (0, 4, 7, 10): 'dominant7',
    (0, 4, 7, 11): 'maj7',
    (0, 3, 7, 10): 'min7',
    (0, 3, 6, 10): 'ø7',
    (0, 3, 6, 9): 'dim7',
}

def detect_chord_pattern(notes):
    pcs = sorted({n % 12 for n in notes})
    for root in pcs:
        intervals = tuple(sorted((pc - root) % 12 for pc in pcs))
        if intervals in CHORD_PATTERNS:
            return CHORD_PATTERNS[intervals]
    return 'unknown'

def midi_to_chords(mid_file):
    midi = mido.MidiFile(mid_file)
    events = []
    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((abs_time, msg.note, 'on'))
            elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                events.append((abs_time, msg.note, 'off'))
    
    # Sort by absolute time
    events.sort(key=lambda x: x[0])

    active_notes = set()
    chords = []
    current_chord = None
    chord_start = None
    previous_active_notes = None

    # Process events grouped by timestamp
    for t, group in groupby(events, key=lambda x: x[0]):
        group = list(group)

        previous_active_notes = active_notes.copy()

        # First, process all note-offs at this timestamp
        for _, note, typ in group:
            if typ == 'off':
                active_notes.discard(note)

        # Then, process all note-ons at this timestamp
        for _, note, typ in group:
            if typ == 'on':
                active_notes.add(note)

        # Detect chord after processing all simultaneous events
        chord_notes = {n % 12 for n in active_notes}
        chord_pattern = detect_chord_pattern(active_notes) if len(chord_notes) >= 2 else None

        if current_chord is None and chord_pattern:
            # Start a new chord
            current_chord = chord_pattern
            chord_start = t
        elif current_chord:
            # Check if the chord changed or ended
            prev_notes = {n % 12 for n in chords[-1]['notes']} if chords else set()
            if chord_pattern != current_chord or chord_notes != prev_notes:
                # End previous chord
                chords.append({
                    'pattern': current_chord,
                    'notes': sorted(previous_active_notes),
                    'start': chord_start,
                    'end': t
                })
                # Start new chord if applicable
                if chord_pattern:
                    current_chord = chord_pattern
                    chord_start = t
                else:
                    current_chord = None
                    chord_start = None

    # Handle the last chord
    if current_chord and chord_start is not None:
        last_time = events[-1][0]
        chords.append({
            'pattern': current_chord,
            'notes': sorted(previous_active_notes),
            'start': chord_start,
            'end': last_time
        })

    return chords

def _print_chords(chords):
    for c in chords:
        print(f"{c['pattern']} chord from {c['start']} to {c['end']}, notes={c['notes']}")

# Example usage
# mid_file = "generated.mid"
# chords = midi_to_chords(mid_file)
# print(len(chords))
# print(chords)
