"""Operations for implementing binary neurons using PyTorch. Code adapted from TensorFlow version to use PyTorch autograd and straight-through estimators."""
import torch

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _BernoulliSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prob):
        return (prob > torch.rand_like(prob)).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _SigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _BernoulliReinforceSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        prob = input.sigmoid()
        sample = (prob > torch.rand_like(prob)).float()
        ctx.save_for_backward(sample, prob)
        return sample

    @staticmethod
    def backward(ctx, grad_output):
        sample, prob = ctx.saved_tensors
        return grad_output * (sample - prob)

# Straight-through round
binary_round = _RoundSTE.apply

# Bernoulli sample with identity gradient
bernoulli_sample = _BernoulliSTE.apply

# Sigmoid with identity gradient
pass_through_sigmoid = _SigmoidSTE.apply

# Binary stochastic neuron using straight-through estimator
# Returns tuple (sample, probability)
def binary_stochastic_ST(x, slope=1.0, pass_through=True, stochastic=True):
    if pass_through:
        p = pass_through_sigmoid(x)
    else:
        p = torch.sigmoid(slope * x)
    if stochastic:
        return bernoulli_sample(p), p
    else:
        return binary_round(p), p

# Binary stochastic neuron using REINFORCE estimator
# Returns sample only; probability can be obtained via torch.sigmoid(x)
def binary_stochastic_REINFORCE(x):
    return _BernoulliReinforceSTE.apply(x)

# Wrapper to apply binary neurons on logits
# estimator: 'straight_through' or 'reinforce'
# stochastic: bool for sampling vs deterministic
# pass_through and slope apply only for 'straight_through'
def binary_wrapper(
    pre_activations,
    estimator,
    stochastic=True,
    pass_through=True,
    slope=1.0
):
    if estimator == 'straight_through':
        sample, p = binary_stochastic_ST(
            pre_activations, slope, pass_through, stochastic
        )
        return sample, p
    elif estimator == 'reinforce':
        if stochastic:
            sample = binary_stochastic_REINFORCE(pre_activations)
        else:
            sample = binary_round(pass_through_sigmoid(pre_activations))
        return sample, pre_activations.sigmoid()
    else:
        raise ValueError(f"Unrecognized estimator '{estimator}'")
