from .faces import FaceCompare, MostSimilar

NODE_CLASS_MAPPINGS = { 
    "Face Compare" : FaceCompare,
    "Most Similar" : MostSimilar,
}

__all__ = ['NODE_CLASS_MAPPINGS']