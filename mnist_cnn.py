"""Guía 05 - Parte 1: CNN para clasificación de dígitos MNIST.

Respeta la arquitectura, las 5000 muestras y la única época de la guía.
Guarda evidencias reproducibles para el análisis solicitado en README.md.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

RUTA_PROYECTO = Path(__file__).resolve().parent
EVIDENCIAS = RUTA_PROYECTO / "evidencias"
RESULTADOS = RUTA_PROYECTO / "resultados"
EVIDENCIAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEED = 42
MUESTRAS = 5000
EPOCAS = 1
BATCH_SIZE = 64


def main():
    tf.keras.utils.set_random_seed(SEED)
    np.set_printoptions(precision=6, suppress=True)
    print("\n--- [PARTE 1] INICIO DE EJECUCIÓN: MNIST ---")
    print(f"TensorFlow: {tf.__version__}")

    # 1. Cargar el dataset original de imágenes (28 x 28, escala de grises).
    (x_train_original, y_train), (x_test_original, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )
    print(f"Forma de los datos originales: {x_train_original.shape}")
    print(f"Rango de píxeles original: {x_train_original.min()} a {x_train_original.max()}")

    # ACTIVIDAD 1: observar 9 imágenes ANTES de normalizar.
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train_original[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Etiqueta: {y_train[i]}")
        ax.axis("off")
    fig.suptitle("MNIST: imágenes originales de entrenamiento", fontsize=13)
    fig.tight_layout()
    destino = EVIDENCIAS / "mnist_muestras.png"
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    print(f"Gráfica guardada: {destino}")
    print("Cierra la ventana de las 9 imágenes para continuar el entrenamiento.")
    plt.show()
    plt.close(fig)

    # Normalizar intensidades [0, 255] -> [0, 1] y agregar canal de color.
    x_train = np.expand_dims(x_train_original.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test_original.astype("float32") / 255.0, axis=-1)
    print(f"Forma para entrada de CNN: {x_train.shape}; prueba: {x_test.shape}")

    # 2. Arquitectura solicitada: Conv32 -> Pool -> Conv64 -> Pool -> Flatten -> Dense64 -> Softmax10.
    model = models.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", name="Conv_1"),
            layers.MaxPooling2D((2, 2), name="MaxPool_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="Conv_2"),
            layers.MaxPooling2D((2, 2), name="MaxPool_2"),
            layers.Flatten(name="Aplanado_Flatten"),
            layers.Dense(64, activation="relu", name="Capa_Densa_Oculta"),
            layers.Dense(10, activation="softmax", name="Salida_Softmax"),
        ],
        name="CNN_MNIST",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n--- RESUMEN DE ARQUITECTURA MNIST ---")
    model.summary()
    print(f"Parámetros entrenables: {model.count_params():,}")
    assert model.count_params() == 121930, "El número de parámetros no coincide con la guía."

    # 3. Única época con 5000 imágenes, como solicita el docente.
    print("\n--- ENTRENANDO MNIST: 5000 imágenes, 1 época ---")
    history = model.fit(
        x_train[:MUESTRAS], y_train[:MUESTRAS],
        epochs=EPOCAS, batch_size=BATCH_SIZE, verbose=1,
    )
    train_loss = float(history.history["loss"][-1])
    train_accuracy = float(history.history["accuracy"][-1])

    # Evaluación adicional sobre 10 000 imágenes NO usadas en entrenamiento.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nPérdida entrenamiento: {train_loss:.6f}")
    print(f"Precisión entrenamiento: {train_accuracy:.4%}")
    print(f"Pérdida conjunto de prueba: {test_loss:.6f}")
    print(f"Precisión conjunto de prueba: {test_accuracy:.4%}")

    # 4. Una predicción con la primera imagen de prueba.
    imagen_prueba = x_test[0:1]
    vector = model.predict(imagen_prueba, verbose=0)[0]
    clase_predicha = int(np.argmax(vector))
    clase_real = int(y_test[0])
    suma = float(np.sum(vector))  # Con todos los decimales, no la versión redondeada.
    confianza = float(vector[clase_predicha])
    print("\n--- PREDICCIÓN MNIST Y ANÁLISIS SOFTMAX ---")
    print("Vector de 10 probabilidades Softmax (sin redondeo a 4 decimales):")
    print(vector)
    print(f"Suma del vector (precisión completa): {suma:.8f}")
    print(f"Índice de mayor probabilidad (np.argmax): {clase_predicha}")
    print(f"Confianza Softmax máxima: {confianza:.4%}")
    print(f"Clase real: {clase_real}")
    print(f"¿Predicción correcta?: {'SÍ' if clase_predicha == clase_real else 'NO'}")

    # Evidencia gráfica: predicción y distribución Softmax de una muestra.
    fig, (ax_img, ax_prob) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_img.imshow(x_test_original[0], cmap="gray", vmin=0, vmax=255)
    ax_img.set_title(f"Real: {clase_real} | Predicho: {clase_predicha}")
    ax_img.axis("off")
    ax_prob.bar(np.arange(10), vector)
    ax_prob.set_xticks(np.arange(10))
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel("Dígito (clase)")
    ax_prob.set_ylabel("Probabilidad Softmax")
    ax_prob.set_title("Vector Softmax de 10 clases")
    fig.tight_layout()
    destino_pred = EVIDENCIAS / "mnist_prediccion.png"
    fig.savefig(destino_pred, dpi=160, bbox_inches="tight")
    print(f"Gráfica guardada: {destino_pred}")
    print("Cierra la ventana de predicción para finalizar.")
    plt.show()
    plt.close(fig)

    # Guardar datos REALES para actualizar el README y generar el contraste.
    informacion = {
        "dataset": "MNIST",
        "seed": SEED,
        "muestras_entrenamiento": MUESTRAS,
        "muestras_prueba": len(y_test),
        "epocas": EPOCAS,
        "batch_size": BATCH_SIZE,
        "parametros": model.count_params(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "probabilidades": [float(p) for p in vector],
        "suma_probabilidades": suma,
        "clase_predicha": clase_predicha,
        "clase_real": clase_real,
        "etiqueta_predicha": str(clase_predicha),
        "etiqueta_real": str(clase_real),
        "confianza": confianza,
        "correcta": bool(clase_predicha == clase_real),
    }
    ruta = RESULTADOS / "mnist_resultados.json"
    ruta.write_text(json.dumps(informacion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados reales guardados en: {ruta}")
    print("--- [PARTE 1] FINALIZADA CORRECTAMENTE ---\n")


if __name__ == "__main__":
    main()
