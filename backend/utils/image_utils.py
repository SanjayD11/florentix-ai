from PIL import Image
import numpy as np
import io

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes):
    """
    Takes raw image bytes and returns a preprocessed numpy array
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # normalize

    image_array = np.expand_dims(image_array, axis=0)
    return image_array
