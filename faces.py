from deepface import DeepFace
import numpy as np
import torch
import folder_paths
from PIL import Image
import os, random

DETECTORS = [ 'ssd', 'mtcnn', 'retinaface', 'mediapipe', 'opencv',  ]
# 'dlib' seems not to download  (issue #2), 'yolov8','yunet','fastmtcnn' detect nothing (issue #3)
# 'opencv' doesn't detect all faces (issue #6)

MODELS = ["Facenet512", "Facenet", "VGG-Face", "OpenFace", "DeepFace", "Dlib"]
# "DeepID" gives everything 0.0 (issue #5)

def save_temp(image:torch.Tensor):
    filepath = os.path.join(folder_paths.get_temp_directory(),f"{random.randint(1000000,9999999)}.png")
    image = 255. * image.cpu().numpy()
    image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    image.save(filepath)
    return filepath

def similarity(file1, file2, detector, model):
    try:
        result = DeepFace.verify(img1_path = file1, img2_path = file2, detector_backend=detector, model_name=model)
        return 1.0 - result['distance']
    except ValueError:
        return 0.0

class FaceCompare:
    RETURN_TYPES = ("FLOAT","STRING",)
    RETURN_NAMES = ("similarity","message",)
    FUNCTION = "func"
    CATEGORY = "faces"
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
            "image1" : ("IMAGE", {}), 
            "image2" : ("IMAGE", {}),  
            "detector": (DETECTORS, {}),
            "model" : (MODELS, {}),
            }, }
    
    def func(self, image1, image2, detector, model):
        if not image1.shape[0]==1 and image2.shape[0]==1: return (0,"Batches not supported",)
        f1 = save_temp(image1[0])
        f2 = save_temp(image2[0])
        s = similarity(f1, f2, detector, model)
        return (s,f"{s}",)
    
class MostSimilar:
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("ordered","message",)
    FUNCTION = "func"
    CATEGORY = "faces"
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
            "true_image" : ("IMAGE", {}), 
            "candidates" : ("IMAGE", {}),
            "detector": (DETECTORS, {}),   
            "model" : (MODELS, {}), 
            }, }
    
    
    def func(self, true_image:torch.Tensor, candidates:torch.Tensor, detector, model):
        if not true_image.shape[0]==1: return (None,0,"Need a single true_image",)
        f1 = save_temp(true_image[0])

        similarities = [(similarity(f1,save_temp(i2),detector,model),i2) for i2 in candidates]
        similarities.sort(reverse=True, key=lambda a:a[0])
        
        return (torch.stack(tuple([s[1] for s in similarities])), f"{tuple([s[0] for s in similarities])}", )
        