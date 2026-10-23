from .scooby import Scooby


def __getattr__(name):
    # Lazy, so importing the Borzoi model does not load alphagenome-pytorch.
    if name == "ScoobyAlphaGenome":
        from ..scag.model import ScoobyAlphaGenome

        return ScoobyAlphaGenome
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
