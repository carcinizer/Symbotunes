import pypianoroll
import numpy as np
from pypianoroll import Track, Multitrack
import torch

class PianorollToMidiConverter:
    """
    Convert a pianoroll tensor or numpy array to a MIDI file.
    """
    def __init__(
        self,
        programs: list[int],
        is_drums: list[bool],
        tempo: int,
        beat_resolution: int,
        lowest_pitch: int,
    ) -> None:
        self.programs = programs
        self.is_drums = is_drums
        self.tempo = tempo
        self.beat_resolution = beat_resolution
        self.lowest_pitch = lowest_pitch

    def __call__(self, pr_input, filename: str):
        # Convert to numpy
        if isinstance(pr_input, torch.Tensor):
            pr = pr_input.detach().cpu().numpy()
        else:
            pr = np.array(pr_input)
        # Expect pr shape: (time, pitch, tracks)
        # Create Track objects
        tracks = []
        for idx in range(pr.shape[2]):
            track_pr = pr[..., idx]
            # pad to full pitch range if needed
            # Here we assume track_pr already includes lowest_pitch offset
            t = Track(pianoroll=track_pr, program=self.programs[idx], is_drum=self.is_drums[idx])
            tracks.append(t)
        mt = Multitrack(tracks=tracks, tempo=self.tempo, beat_resolution=self.beat_resolution)
        # Save as MIDI if filename ends with .mid
        mt.write(filename)
