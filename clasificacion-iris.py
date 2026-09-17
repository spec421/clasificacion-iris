# -*- coding: utf-8 -*-
# Requiere: tensorflow, scikit-learn, matplotlib, numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Cargar datos
iris = load_iris()
X = iris['data'][:, [2, 3]]  # petal length, petal width
y = iris['target']
class_names = iris['target_names']

# 2. Train/Test split
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Modelo
model = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation='relu'),
    #layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')

])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    #callbacks=[es],
    verbose=0
)

# 6. Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en test: {test_acc:.4f}\n")

# 7. Predicciones
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=class_names))

# 8. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Matriz de confusión - Red neuronal")
plt.show()

# 9. Curvas de entrenamiento (loss y accuracy)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axs[0].plot(history.history['loss'], label='Entrenamiento')
axs[0].plot(history.history['val_loss'], label='Validación')
axs[0].set_title('Curva de pérdida')
axs[0].set_xlabel('Épocas')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Accuracy
axs[1].plot(history.history['accuracy'], label='Entrenamiento')
axs[1].plot(history.history['val_accuracy'], label='Validación')
axs[1].set_title('Curva de exactitud')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.show()


# 4. Crear una malla de puntos (rejilla)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

# 5. Predecir la clase en cada punto de la malla
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1).reshape(xx.shape)

# 6. Visualizar fronteras de decisión
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Puntos del conjunto de prueba (test set)
for i, class_name in enumerate(class_names):
    plt.scatter(
        X_test[y_test==i, 0], X_test[y_test==i, 1],
        label=class_name, edgecolor='k', s=60
    )

plt.title("Fronteras de decisión - Red neuronal (Iris) con datos de prueba")
plt.xlabel("Petal length (estandarizado)")
plt.ylabel("Petal width (estandarizado)")
plt.legend()
plt.show()
