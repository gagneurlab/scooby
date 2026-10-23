"""scooby: modeling multi-modal genomic profiles from DNA sequence at single-cell resolution."""

from importlib import import_module

# Imported on first access, so `import scooby` does not load torch.
_LAZY_EXPORTS = {
    "ScoobyAlphaGenome": "scooby.scag.model",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
