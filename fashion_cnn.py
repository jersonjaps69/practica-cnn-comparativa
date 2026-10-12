"""Guía 05 - Parte 2: CNN para clasificación de prendas Fashion-MNIST.

Respeta el enunciado: 5000 imágenes, 1 época, capa densa de 128 neuronas.
Incluye mapeo nominal de categorías y evidencias para el README.
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
CLASS_NAMES = [
    "Camiseta/Top", "Pantalón", "Pulóver", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]


def main():
    tf.keras.utils.set_random_seed(SEED)
    np.set_printoptions(precision=6, suppress=True)
    print("\n--- [PARTE 2] INICIO DE EJECUCIÓN: FASHION-MNIST ---")
    print(f"TensorFlow: {tf.__version__}")

    # 1. Cargar dataset Fashion-MNIST en escala de grises.
    (x_train_original, y_train), (x_test_original, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    print(f"Forma de los datos originales: {x_train_original.shape}")
    print(f"Rango de píxeles original: {x_train_original.min()} a {x_train_original.max()}")

    # ACTIVIDAD 1: mostrar y guardar 9 siluetas ANTES de normalizar.
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train_original[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(CLASS_NAMES[int(y_train[i])], fontsize=10)
        ax.axis("off")
    fig.suptitle("Fashion-MNIST: imágenes originales de entrenamiento", fontsize=13)
    fig.tight_layout()
    destino = EVIDENCIAS / "fashion_muestras.png"
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    print(f"Gráfica guardada: {destino}")
    print("Cierra la ventana de las 9 prendas para continuar el entrenamiento.")
    plt.show()
    plt.close(fig)

    # Imágenes de 28 x 28 x 1 y píxeles normalizados entre 0 y 1.
    x_train = np.expand_dims(x_train_original.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test_original.astype("float32") / 255.0, axis=-1)
    print(f"Forma para entrada de CNN: {x_train.shape}; prueba: {x_test.shape}")

    # 2. Diferencia central con MNIST: capa densa de 128 (no de 64).
    model = models.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", name="Conv_1"),
            layers.MaxPooling2D((2, 2), name="MaxPool_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="Conv_2"),
            layers.MaxPooling2D((2, 2), name="MaxPool_2"),
            layers.Flatten(name="Aplanado_Flatten"),
            layers.Dense(128, activation="relu", name="Capa_Densa_Oculta"),
            layers.Dense(10, activation="softmax", name="Salida_Softmax"),
        ],
        name="CNN_Fashion_MNIST",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n--- RESUMEN DE ARQUITECTURA FASHION-MNIST ---")
    model.summary()
    print(f"Parámetros entrenables: {model.count_params():,}")
    assert model.count_params() == 225034, "El número de parámetros no coincide con el esperado."

    # 3. Entrenar con condiciones solicitadas.
    print("\n--- ENTRENANDO FASHION-MNIST: 5000 imágenes, 1 época ---")
    history = model.fit(
        x_train[:MUESTRAS], y_train[:MUESTRAS],
        epochs=EPOCAS, batch_size=BATCH_SIZE, verbose=1,
    )
    train_loss = float(history.history["loss"][-1])
    train_accuracy = float(history.history["accuracy"][-1])

    # Evaluación independiente: no reemplaza ni altera la época pedida.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nPérdida entrenamiento: {train_loss:.6f}")
    print(f"Precisión entrenamiento: {train_accuracy:.4%}")
    print(f"Pérdida conjunto de prueba: {test_loss:.6f}")
    print(f"Precisión conjunto de prueba: {test_accuracy:.4%}")

    # 4. Clasificación de la primera imagen de prueba.
    vector = model.predict(x_test[0:1], verbose=0)[0]
    clase_predicha = int(np.argmax(vector))
    clase_real = int(y_test[0])
    suma = float(np.sum(vector))
    confianza = float(vector[clase_predicha])
    print("\n--- PREDICCIÓN FASHION-MNIST Y ANÁLISIS SOFTMAX ---")
    print("Vector de 10 probabilidades Softmax (sin redondeo a 4 decimales):")
    print(vector)
    print(f"Suma del vector (precisión completa): {suma:.8f}")
    print(f"Índice de mayor probabilidad (np.argmax): {clase_predicha}")
    print(f"Clase predicha: {CLASS_NAMES[clase_predicha]} (índice {clase_predicha})")
    print(f"Clase real: {CLASS_NAMES[clase_real]} (índice {clase_real})")
    print(f"Confianza Softmax máxima: {confianza:.4%}")
    print(f"¿Predicción correcta?: {'SÍ' if clase_predicha == clase_real else 'NO'}")

    # Evidencia: muestra y probabilidades para todas las categorías.
    fig, (ax_img, ax_prob) = plt.subplots(1, 2, figsize=(13.5, 5))
    ax_img.imshow(x_test_original[0], cmap="gray", vmin=0, vmax=255)
    ax_img.set_title(f"Real: {CLASS_NAMES[clase_real]}\nPredicho: {CLASS_NAMES[clase_predicha]}")
    ax_img.axis("off")
    ax_prob.bar(np.arange(10), vector)
    ax_prob.set_xticks(np.arange(10), CLASS_NAMES, rotation=55, ha="right")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel("Clase de prenda")
    ax_prob.set_ylabel("Probabilidad Softmax")
    ax_prob.set_title("Distribución de probabilidades de vestimenta")
    fig.tight_layout()
    destino_pred = EVIDENCIAS / "fashion_prediccion.png"
    fig.savefig(destino_pred, dpi=160, bbox_inches="tight")
    print(f"Gráfica guardada: {destino_pred}")
    print("Cierra la ventana de predicción para finalizar.")
    plt.show()
    plt.close(fig)

    informacion = {
        "dataset": "Fashion-MNIST",
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
        "etiqueta_predicha": CLASS_NAMES[clase_predicha],
        "etiqueta_real": CLASS_NAMES[clase_real],
        "confianza": confianza,
        "correcta": bool(clase_predicha == clase_real),
    }
    ruta = RESULTADOS / "fashion_resultados.json"
    ruta.write_text(json.dumps(informacion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados reales guardados en: {ruta}")
    print("--- [PARTE 2] FINALIZADA CORRECTAMENTE ---\n")


if __name__ == "__main__":
    main()
