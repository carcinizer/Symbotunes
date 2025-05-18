
from music21 import converter, exceptions21
from music21.midi.translate import music21ObjectToMidiFile
from io import BytesIO
from random import choice
import symusic


class RandomTranspose:
    def __init__(self, half_steps: list[int]):
        self.half_steps = half_steps

    def __call__(self, data: bytes) -> bytes:
        try:
            stream = converter.parse(data, format="midi")
        except exceptions21.StreamException: # Empty stream
            return data
        io = BytesIO()
        half_steps = choice(self.half_steps)
        stream.transpose(half_steps)

        midi = music21ObjectToMidiFile(stream)
            
        midi.openFileLike(io)
        midi.write()
        out = io.getvalue()
        midi.close()
        return out


class RandomTimestretch:
    def __init__(self, magnitudes: list[float]):
        self.magnitudes = magnitudes

    def __call__(self, data: bytes) -> bytes:
        score_sec = symusic.Score.from_midi(data, ttype="second")
        mag = choice(self.magnitudes)

        for track in score_sec.tracks:
            for note in track.notes:
                note.start *= mag
                note.duration *= mag

        return score_sec.dumps_midi()


