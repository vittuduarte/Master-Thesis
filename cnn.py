# 1. Importação das Bibliotecas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPool1D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import preprocessing
from sklearn.metrics import classification_report
import zipfile

# 2. Pré-processamento e Extração dos Arquivos
file_name = "ECG200.zip"

with zipfile.ZipFile(file_name, 'r') as zipf:
    print('Extração dos arquivos....')
    zipf.extractall()
    print('Finalizado!')

# 3. Carregamento da Base de Dados
data_train = arff.loadarff('ECG200_TRAIN.arff')
df_train = pd.DataFrame(data_train[0])

data_test = arff.loadarff('ECG200_TEST.arff')
df_test = pd.DataFrame(data_test[0])

# 4. Label Encoder para a variável Target
le = preprocessing.LabelEncoder()
df_train['target'] = le.fit_transform(df_train['target'])
df_test['target'] = le.transform(df_test['target'])

# 5. Separação de X (Features) e Y (Targets)
X_train = df_train.drop('target', axis=1).values
y_train = df_train['target'].values

X_test = df_test.drop('target', axis=1).values
y_test = df_test['target'].values

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)

# ==============================================================================
# 6. Modelo CNN 1D
# ==============================================================================
batch_size = 64
epochs = 20
num_classes = len(np.unique(y_train))

model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool1D(padding='same'))

model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool1D(padding='same'))

model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool1D(padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

# 7. Compilação do Modelo
# sparse_categorical_crossentropy é mais apropriado quando os labels são inteiros (0, 1).
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()

# 8. Treinamento do Modelo
model_train = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test))

# 9. Avaliação e Gráficos
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

accuracy = model_train.history['accuracy']
val_accuracy = model_train.history['val_accuracy']
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs_range = range(len(accuracy))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# ==============================================================================
# 10. CNN Com Dropout
# ==============================================================================

# O nome da variável não pode começar com números
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28, 28, 1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.4))

fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(10, activation='softmax')) # num_classes = 10 para Fashion MNIST

# Salvar Modelo
model.save("model_dropout.h5")


"""
# Rótulos de previsão e classificação correta/incorreta
predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

# Classificação correta
correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

# Classificação incorreta
incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

# Relatório de Classificação
target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
"""