"""BandTok: modeling music as a time-frequency image.

A 2D Mel-spectrogram tokenizer and an autoregressive music language model over
flattened time-frequency tokens.
"""

__version__ = "0.1.0"

__all__ = [
    "BandTok",
    "BandTokLM",
    "create_model_from_config",
    "create_tokenizer_from_config",
    "load_model_config",
]


def __getattr__(name):
    # Imported lazily so that `import bandtok` does not pay the torch/transformers
    # import cost until a model is actually requested.
    if name == "BandTok":
        from .models.tokenizer import BandTok

        return BandTok
    if name == "BandTokLM":
        from .models.lm import BandTokLM

        return BandTokLM
    if name in ("create_model_from_config", "create_tokenizer_from_config"):
        from .models import factory

        return getattr(factory, name)
    if name == "load_model_config":
        from .config import load_model_config

        return load_model_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
