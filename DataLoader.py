import cv2
import numpy as np

class DataLoader:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def load_image(self, image_path):
        """Lädt ein einzelnes Bild und bereitet es vor."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = np.expand_dims(img, axis=-1)  # Dimensionalität anpassen
        img = img / 255.0  # Normalisierung
        return img

    def load_dataset(self, image_paths, labels):
        """Lädt mehrere Bilder und ihre Labels."""
        images = [self.load_image(img_path) for img_path in image_paths]
        return np.array(images), np.array(labels)
