
from miditok import REMI, TSD, MIDILike, TokenizerConfig, TokSequence, CPWord
import symusic
import torch


TOKENIZER_PARAMS = {
    "pitch_range": (0, 127),
    "vocab_size": 130,  # Total vocab size: 130 for pitches, note-off, rest + 512 for drum patterns
    "max_bar_length": 16,  # Each bar has 16 events (16th notes)
    "num_bars": 16,  # Total number of bars for hierarchical model
    "hierarchical": True,  # Use hierarchical modeling
    "bar_token_count": 16,  # Each subsequence corresponds to a single bar
    "beat_resolution": 4,  # 16th note intervals (4 intervals per beat in 4/4 time)
    "note_on_count": 128,  # 128 note-on tokens
    # "use_rests": True,
    # "drum_pattern_count": 512  # 512 categorical tokens for drum patterns
}

class MidiTokTokenizer(object):
    def __init__(self, tokenizer_params: dict, max_tracks: int = 1, input_format: str = "midi") -> None:
        self.config = TokenizerConfig(**tokenizer_params)
        match tokenizer_params.get("tokenization", "remi"):
            case "remi":
                self.tokenizer = REMI(self.config)
            case "remiplus":
                self.config.use_programs = True
                self.config.one_token_stream_for_programs = True
                self.config.use_time_signatures = True
                self.tokenizer = REMI(self.config)
            case "tsd":
                self.tokenizer = TSD(self.config)
            case "midilike":
                self.tokenizer = MIDILike(self.config)
            case "cpword":
                self.tokenizer = CPWord(self.config)
            case _:
                raise Exception(f"Unknown tokenization '{tokenizer_params.get('tokenization')}'")

        self.input_format = input_format

        self.max_tracks = max_tracks

    def __call__(self, data: bytes):
        try:
            match self.input_format:
                case "midi":
                    score = symusic.Score.from_midi(data)
                case "abc":
                    score = symusic.Score.from_abc(str(data))
                case _:
                    raise Exception(f"Unknown format '{self.input_format}'")
        except Exception as e:
            raise Exception(f"Failed to tokenize: {e}, data: {data}")

        tokenized_midi = self.tokenizer(score)[: self.max_tracks]
        return tokenized_midi[0] if len(tokenized_midi) == 1 else tokenized_midi

    def inverse_transform(self, data: TokSequence) -> bytes:
        return self.tokenizer.decode(data).dumps_midi()



class MusicVAETokenizer(MidiTokTokenizer):
    def __init__(self) -> None:
        super().__init__(TOKENIZER_PARAMS)
