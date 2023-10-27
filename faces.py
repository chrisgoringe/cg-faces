import face_recognition
import numpy as np
import torch
import folder_paths
from PIL import Image
import os

def convert(image:torch.Tensor):
    filepath = os.path.join(folder_paths.get_temp_directory(),"fr.png")
    image = 255. * image.cpu().numpy()
    image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    image.save(filepath)
    return face_recognition.load_image_file(filepath)

class FaceCompare:
    RETURN_TYPES = ("FLOAT","STRING",)
    RETURN_NAMES = ("similarity","message",)
    FUNCTION = "func"
    CATEGORY = "faces"
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image1" : ("IMAGE", {}), "image2" : ("IMAGE", {}),  }, }
    
    def func(self, image1, image2):
        if not image1.shape[0]==1 and image2.shape[0]==1: return (0,"Batches not supported",)
        f1 = face_recognition.face_encodings(convert(image1[0]), model="large")
        f2 = face_recognition.face_encodings(convert(image2[0]), model="large")

        if not (len(f1)==1 and len(f2)==1): return (0,"Need one face in each image",)

        s = 1.0 - face_recognition.face_distance(f1, f2[0])[0]
        return (s,f"{s}",)
    
class MostSimilar:
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("ordered","message",)
    FUNCTION = "func"
    CATEGORY = "faces"
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "true_image" : ("IMAGE", {}), "candidates" : ("IMAGE", {}),  }, }
    
    def func(self, true_image, candidates):
        if not true_image.shape[0]==1: return (None,0,"Need a single true_image",)
        f1 = face_recognition.face_encodings(convert(true_image[0]), model="large")
        if len(f1)!=1: return(None,0,"Need exactly one face in true_image")
        similarities = []
        for i2 in candidates:
            con = convert(i2)
            f2 = face_recognition.face_encodings(con, model="large")
            if len(f2)==1:
                similarities.append((1.0-face_recognition.face_distance(f1, f2[0])[0],i2))
            else:
                similarities.append((0,i2))
        
        similarities.sort(reverse=True)
        
        return (torch.stack(tuple([s[1] for s in similarities])), f"{tuple([s[0] for s in similarities])}", )
        