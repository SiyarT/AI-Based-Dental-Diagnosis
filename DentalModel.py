from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class DentalModel:
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):
        """Erstellt und kompiliert das Modell."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binäre Klassifikation
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Trainiert das Modell."""
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        return history

    def evaluate(self, X_test, y_test):
        """Bewertet das Modell."""
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        return test_acc
