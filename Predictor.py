class Predictor:
    def __init__(self, model):
        self.model = model

    def predict_image(self, image_path):
        """Macht eine Vorhersage für ein einzelnes Bild."""
        img = DataLoader().load_image(image_path)
        img = np.expand_dims(img, axis=0)  # Dimensionalität anpassen
        prediction = self.model.predict(img)
        return prediction

    def get_prediction_label(self, image_path):
        """Gibt die Vorhersage und den Diagnose-Label zurück."""
        prediction = self.predict_image(image_path)
        if prediction >= 0.5:
            return "Problem detected"
        else:
            return "No problem detected"
