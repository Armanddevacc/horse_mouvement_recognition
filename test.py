import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Création d'un modèle simple pour tester l'importation
model = Sequential()
model.add(LSTM(10, input_shape=(100, 12)))  # Exemple de modèle avec LSTM
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy')

print("TensorFlow fonctionne correctement avec LSTM et Dense.")
