import pypianoroll
import numpy as np
import torch
import pretty_midi

class MidiToPianorollTransform:
    """
    Transform MIDI file or pypianoroll.Multitrack to a pianoroll tensor.
    """
    def __init__(
        self,
        n_time_steps: int,
        lowest_pitch: int,
        n_pitches: int,
        n_tracks: int,
    ) -> None:
        self.n_time_steps = n_time_steps
        self.lowest_pitch = lowest_pitch
        self.n_pitches = n_pitches
        self.n_tracks = n_tracks
        
    def __call__(self, midi_input):
        # Load multitrack pianoroll
        if isinstance(midi_input, str):
            mt = pypianoroll.read(midi_input)
        else:
            mt = midi_input
        # Convert to Multitrack if pypianoroll.read returned a PrettyMIDI
        if isinstance(mt, pretty_midi.PrettyMIDI):
            mt = pypianoroll.from_pretty_midi(mt)
        
        # Extract pianorolls from tracks
        # Each track has a pianoroll attribute
        pianorolls = []
        for track in mt.tracks:
            pianorolls.append(track.pianoroll)
        
        # Stack pianorolls if we have any
        if pianorolls:
            # Stack along the last dimension
            pr = np.stack(pianorolls, axis=2)
            # Transpose if needed - pianoroll is already (time, pitch), we just stacked tracks
            # so no need for transpose
        else:
            # Create an empty pianoroll with the right shape
            pr = np.zeros((0, 128, 0), dtype=np.bool_)
        # Trim or pad time dimension
        T = self.n_time_steps
        if pr.shape[0] >= T:
            pr = pr[:T]
        else:
            pad = np.zeros((T - pr.shape[0], pr.shape[1], pr.shape[2]), dtype=pr.dtype)
            pr = np.concatenate((pr, pad), axis=0)
        # Select pitch range
        start = self.lowest_pitch
        end = start + self.n_pitches
        pr = pr[:, start:end, :]
        # Ensure number of tracks
        if pr.shape[2] > self.n_tracks:
            pr = pr[:, :, : self.n_tracks]
        elif pr.shape[2] < self.n_tracks:
            pad_tracks = np.zeros((pr.shape[0], pr.shape[1], self.n_tracks - pr.shape[2]), dtype=pr.dtype)
            pr = np.concatenate((pr, pad_tracks), axis=2)
        # Binarize
        pr = pr > 0
        return torch.tensor(pr, dtype=torch.float32)
