

class LoadMIDIData(object):
    def __init__(self):
        pass

    def __call__(self, filename: str) -> bytes:
        with open(filename, "rb") as f:
            return f.read()

