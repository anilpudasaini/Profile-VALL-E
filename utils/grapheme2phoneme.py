import argparse
from phonemizer import phonemize
from phonemizer.backend import BACKENDS
from functools import cache

cached_backends = {}
def _get_backend(backend="espeak", punctuation=True, stress=True, strip=True):
    if backend in cached_backends:
        return cached_backends[backend]

    if backend == 'espeak':
        phonemizer = BACKENDS[backend]("en-us", preserve_punctuation=punctuation, with_stress=stress)
    elif backend == 'espeak-mbrola':
        phonemizer = BACKENDS[backend]("en-us")
    else:
        phonemizer = BACKENDS[backend]("en-us", preserve_punctuation=punctuation)

    cached_backends[backend] = phonemizer
    return phonemizer


def encode(text: str, backend="espeak", punctuation=True, stress=True, strip=True) -> list[str]:
    backend = _get_backend(backend=backend, stress=stress, strip=strip, punctuation=punctuation)
    tokens = backend.phonemize([text], strip=strip)

    if not tokens:
        raise Exception(f"Failed to phonemize, received empty string: {text}")

    return tokens[0]


# Helper function to debug phonemizer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("string", type=str)
    parser.add_argument("--backend", type=str, default="espeak")
    parser.add_argument("--no-punctuation", action="store_true")
    parser.add_argument("--no-stress", action="store_true")
    parser.add_argument("--no-strip", action="store_true")

    args = parser.parse_args()

    phonemes = encode(args.string, backend=args.backend, punctuation=not args.no_punctuation, stress=not args.no_stress, strip=not args.no_strip)
    print(phonemes)
