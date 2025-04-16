# from random import randint # TODO should the transform sample the midi sequence at random?
from miditok import TokSequence
from random import randint
import torch



class SampleBars(object):
    """Sample n bars from midi file or less if midi does not have enough bars"""

    def __init__(self, number_of_bars: int) -> None:
        self.num_bars = number_of_bars

    def __call__(self, tokens: TokSequence) -> TokSequence:
        token_names = tokens.tokens
        bar_positions = [i for i, token in enumerate(token_names) if token.startswith("Bar_")]

        if len(bar_positions) < self.num_bars:
            return tokens

        max_start_index = len(bar_positions) - self.num_bars
        start_index = randint(0, max_start_index)

        start_pos = bar_positions[start_index]
        end_pos = (
            bar_positions[start_index + self.num_bars]
            if (start_index + self.num_bars) < len(bar_positions)
            else len(token_names)
        )

        sampled_tokens = tokens[start_pos:end_pos]
        return sampled_tokens


class TokSequenceToTensor(object):
    def __call__(self, tokens: TokSequence) -> torch.Tensor:
        return torch.tensor(tokens.ids)


class TensorToTokSequence(object):
    def __call__(self, tensor: torch.Tensor) -> TokSequence:
        return TokSequence(ids=tensor.tolist())

