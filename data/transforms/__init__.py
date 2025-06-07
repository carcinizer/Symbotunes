from typing import Callable
from torchvision import transforms  # type: ignore[import]
from .tokenize_transforms import TokenizeTransform, ReverseTokenizeTransform
from ..converters import ABCTOMidiConverter, SplitLines
from .file_loaders import LoadMIDIData
from ..tokenizers import MidiTokTokenizer, FolkTokenizer
from .midi_transforms import SampleBars, TokSequenceToTensor, TensorToTokSequence, ForceTempo
from .sample_subsequence import SampleSubsequence
from .augmentations import RandomTimestretch, RandomTranspose


# fmt: off
def get_transform(kwargs: dict | list) -> Callable:
    if isinstance(kwargs, list):
        return transforms.Compose([parse_transform(*next(iter(trans.items()))) for trans in kwargs])
    else:
        return transforms.Compose([parse_transform(n, k) for n, k in kwargs.items()])
# fmt: on


def parse_transform(name: str, kwargs: dict) -> Callable:
    kwargs = kwargs if kwargs is not None else dict()
    match name:
        case "folk_rnn":
            return TokenizeTransform(FolkTokenizer(**kwargs))
        case "folk_rnn_inv":
            return ReverseTokenizeTransform(FolkTokenizer(**kwargs))
        case "abc_midi":
            return ABCTOMidiConverter(**kwargs)
        case "load_midi":
            return LoadMIDIData(**kwargs)
        case "midi_tokenizer":
            return TokenizeTransform(MidiTokTokenizer(**kwargs))
        case "midi_tokenizer_inv":
            return ReverseTokenizeTransform(MidiTokTokenizer(**kwargs))
        #case "music_vae_tokenizer":
            #return MusicVAETokenizer()
        case "sample_bars":
            return SampleBars(**kwargs)
        case "toksequence_to_tensor":
            return TokSequenceToTensor(**kwargs)
        case "tensor_to_toksequence":
            return TensorToTokSequence(**kwargs)
        case "sample_subsequence":
            return SampleSubsequence(**kwargs)
        case "augment_timestretch":
            return RandomTimestretch(**kwargs)
        case "augment_transpose":
            return RandomTranspose(**kwargs)
        case "force_tempo":
            return ForceTempo(**kwargs)
        case "split_lines":
            return SplitLines(**kwargs)
        case _:
            raise NotImplementedError("Unknown transform: " + name)
