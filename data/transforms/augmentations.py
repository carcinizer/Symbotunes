
from music21 import converter
from music21.midi.translate import music21ObjectToMidiFile
from io import BytesIO
from random import choice
import symusic


class RandomTranspose:
    def __init__(self, half_steps: list[int]):
        self.half_steps = half_steps

    def __call__(self, data: bytes) -> bytes:
        stream = converter.parse(data, format="midi")
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

        assert len(score_sec.tracks) == 1

        for note in score_sec.tracks[0].notes:
            note.start *= mag
            note.duration *= mag

        return score_sec.dumps_midi()


