from sklearn.model_selection import train_test_split

class DentalDiagnosisSystem:
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.data_loader = DataLoader()
        self.model = DentalModel()
        self.predictor = Predictor(self.model)

    def prepare_data(self):
        """Bereitet die Daten für das Training vor."""
        X, y = self.data_loader.load_dataset(self.image_paths, self.labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, X_val, y_val):
        """Trainiert das Modell."""
        return self.model.train(X_train, y_train, X_val, y_val)

    def evaluate_model(self, X_test, y_test):
        """Bewertet das Modell."""
        return self.model.evaluate(X_test, y_test)

    def predict(self, image_path):
        """Führt eine Vorhersage für ein Bild durch."""
        return self.predictor.get_prediction_label(image_path)
