
class TokenizeTransform(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        return self.tokenizer(data)


class ReverseTokenizeTransform(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        return self.tokenizer.inverse_transform(data)

