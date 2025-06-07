from typing import List
from music21 import converter
from music21.midi.translate import music21ObjectToMidiFile
from io import BytesIO


class ABCTOMidiConverter:
    def __init__(self, resolution=480, tempo=500000):
        self.resolution = resolution
        self.tempo = tempo

    def _reformat_notes(self, notes: List[str]):
        notes.insert(1, "L: 1/8")
        notes.insert(3, "\n")
        notes.insert(2, "\n")
        notes.insert(1, "\n")
        return notes

    def _convert_abc_to_midi(self, notes_string: str) -> bytes:
        stream = converter.parse(notes_string, format="abc")
        io = BytesIO()

        midi = music21ObjectToMidiFile(stream)
        midi.openFileLike(io)
        midi.write()
        out = io.getvalue()
        midi.close()
        return out

    def __call__(self, notes: list[str]) -> bytes:
        assert notes[0][0] == "M", f"Badly formatted input {notes!r}"
        assert notes[1][0] == "K", f"Badly formatted input {notes!r}"

        formatted_notes = self._reformat_notes(notes)
        notes_string = " ".join(formatted_notes)
        return self._convert_abc_to_midi(notes_string)


class SplitLines:
    def __call__(self, notes) -> list[str]:
        return str(notes).splitlines()
