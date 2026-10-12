"""Genera gráficas comparativas y completa la sección de resultados del README.

Ejecutar ÚNICAMENTE después de mnist_cnn.py y fashion_cnn.py.
No inventa métricas: siempre lee los JSON obtenidos del entrenamiento real.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
CARPETA_RESULTADOS = BASE / "resultados"
CARPETA_EVIDENCIAS = BASE / "evidencias"
README = BASE / "README.md"
MARCADOR_INICIO = "<!-- INICIO_RESULTADOS_REALES -->"
MARCADOR_FIN = "<!-- FIN_RESULTADOS_REALES -->"


def abrir_resultado(nombre):
    archivo = CARPETA_RESULTADOS / f"{nombre}_resultados.json"
    if not archivo.exists():
        raise SystemExit(
            f"Falta {archivo.name}. Primero ejecuta "
            + ("mnist_cnn.py" if nombre == "mnist" else "fashion_cnn.py")
        )
    with archivo.open(encoding="utf-8") as archivo_abierto:
        resultado = json.load(archivo_abierto)
    if len(resultado.get("probabilidades", [])) != 10:
        raise SystemExit(f"Las probabilidades de {archivo.name} no contienen 10 clases.")
    return resultado


def formato_probabilidades(datos):
    return "\n".join(f"| {i} | {p:.6f} |" for i, p in enumerate(datos["probabilidades"]))


def juicio_probabilidades(datos):
    valores = np.asarray(datos["probabilidades"], dtype=float)
    top = np.argsort(valores)[::-1]
    i1, i2 = (int(v) for v in top[:2])
    p1, p2 = (float(valores[i1]), float(valores[i2]))
    diferencia = p1 - p2
    if p1 >= 0.8 and diferencia >= 0.5:
        nivel = "concentrada en una clase (claro máximo)"
    elif p1 >= 0.5 and diferencia >= 0.2:
        nivel = "con una clase dominante, pero con cierta competencia entre opciones"
    else:
        nivel = "relativamente distribuida entre varias clases"
    return (
        f"La distribución está **{nivel}**: la clase {i1} obtuvo "
        f"**{p1:.4%}**, la segunda (clase {i2}) obtuvo **{p2:.4%}** "
        f"y la separación es **{diferencia:.4%}** puntos de probabilidad. "
        "Esto caracteriza esta muestra concreta; no basta para afirmar que todas "
        "las imágenes del dataset tengan igual grado de confianza."
    )


def crear_grafica(uno, dos, metrica, titulo, nombre_archivo, eje_y):
    etiquetas = ["MNIST", "Fashion-MNIST"]
    entrenamientos = [uno[f"train_{metrica}"], dos[f"train_{metrica}"]]
    pruebas = [uno[f"test_{metrica}"], dos[f"test_{metrica}"]]
    posiciones = np.arange(2)
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    barras_train = ax.bar(posiciones - ancho / 2, entrenamientos, ancho, label="Entrenamiento")
    barras_test = ax.bar(posiciones + ancho / 2, pruebas, ancho, label="Prueba")
    ax.set_xticks(posiciones, etiquetas)
    ax.set_ylabel(eje_y)
    ax.set_title(titulo)
    ax.legend()
    if metrica == "accuracy":
        ax.set_ylim(0, 1.05)
        ax.bar_label(barras_train, fmt="%.3f", padding=3)
        ax.bar_label(barras_test, fmt="%.3f", padding=3)
    else:
        ax.bar_label(barras_train, fmt="%.4f", padding=3)
        ax.bar_label(barras_test, fmt="%.4f", padding=3)
    fig.tight_layout()
    CARPETA_EVIDENCIAS.mkdir(exist_ok=True)
    archivo = CARPETA_EVIDENCIAS / nombre_archivo
    fig.savefig(archivo, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return archivo


def bloque_resultados(m, f):
    exactitud_mejor = "MNIST" if m["train_accuracy"] > f["train_accuracy"] else "Fashion-MNIST"
    if abs(m["train_accuracy"] - f["train_accuracy"]) < 1e-8:
        exactitud_mejor = "ninguno (empate)"
    perdida_mejor = "MNIST" if m["train_loss"] < f["train_loss"] else "Fashion-MNIST"
    if abs(m["train_loss"] - f["train_loss"]) < 1e-8:
        perdida_mejor = "ninguno (empate)"
    suma_m, suma_f = (sum(k["probabilidades"]) for k in [m, f])
    if m["confianza"] > f["confianza"]:
        mayor_confianza = "MNIST"
    elif m["confianza"] < f["confianza"]:
        mayor_confianza = "Fashion-MNIST"
    else:
        mayor_confianza = "ninguno: ambas muestras tienen el mismo máximo"
    return f"""### Tabla de resultados obtenidos en VS Code

Estos son **datos reales importados de la ejecución de los dos scripts** (`resultados/*.json`), no valores estimados ni de ejemplo. Se respetaron las 5000 muestras y la única época solicitadas; la evaluación sobre prueba es un complemento.

| Indicador | MNIST | Fashion-MNIST |
|---|---:|---:|
| Imágenes utilizadas para entrenar | {m['muestras_entrenamiento']} | {f['muestras_entrenamiento']} |
| Épocas | {m['epocas']} | {f['epocas']} |
| Batch size | {m['batch_size']} | {f['batch_size']} |
| Parámetros entrenables | {m['parametros']:,} | {f['parametros']:,} |
| Pérdida entrenamiento (loss) | {m['train_loss']:.6f} | {f['train_loss']:.6f} |
| Exactitud entrenamiento (accuracy) | {m['train_accuracy']:.4%} | {f['train_accuracy']:.4%} |
| Pérdida prueba (loss) | {m['test_loss']:.6f} | {f['test_loss']:.6f} |
| Exactitud prueba (accuracy) | {m['test_accuracy']:.4%} | {f['test_accuracy']:.4%} |

**Interpretación de loss y accuracy:** en esta ejecución, **{exactitud_mejor}** consiguió el accuracy de entrenamiento más alto y **{perdida_mejor}** consiguió el loss de entrenamiento más bajo. `Accuracy` mide la fracción de predicciones correctas; `loss` cuantifica el error según la pérdida de entropía cruzada. Que un modelo tenga mejor métrica tras una sola época **no demuestra** que converja más rápido a lo largo del tiempo. Además, la capa densa difiere (64 frente a 128), por lo que no se puede atribuir toda la diferencia únicamente a la complejidad de las imágenes.

![Accuracy: entrenamiento y prueba](evidencias/comparacion_accuracy.png)

![Loss: entrenamiento y prueba](evidencias/comparacion_loss.png)

### Predicción de la primera imagen de prueba: MNIST

- **Etiqueta verdadera (`y_test[0]`):** {m['etiqueta_real']} (índice {m['clase_real']}).
- **`np.argmax`:** {m['clase_predicha']}; clase elegida: **{m['etiqueta_predicha']}**.
- **Resultado:** {'acertó' if m['correcta'] else 'no acertó'}.
- **Suma completa del vector Softmax:** {suma_m:.9f} (aproximadamente 1, por definición y precisión numérica).

| Índice de la clase | Probabilidad Softmax (6 decimales) |
|---:|---:|
{formato_probabilidades(m)}

{juicio_probabilidades(m)}

![Predicción y vector de probabilidades MNIST](evidencias/mnist_prediccion.png)

### Predicción de la primera imagen de prueba: Fashion-MNIST

- **Etiqueta verdadera (`y_test[0]`):** {f['etiqueta_real']} (índice {f['clase_real']}).
- **`np.argmax`:** {f['clase_predicha']}; `class_names[{f['clase_predicha']}]` corresponde a **{f['etiqueta_predicha']}**.
- **Resultado:** {'acertó' if f['correcta'] else 'no acertó'}.
- **Suma completa del vector Softmax:** {suma_f:.9f} (aproximadamente 1).

| Índice de la clase | Probabilidad Softmax (6 decimales) |
|---:|---:|
{formato_probabilidades(f)}

{juicio_probabilidades(f)}

![Predicción y vector de probabilidades Fashion-MNIST](evidencias/fashion_prediccion.png)

### Comparación crítica de las probabilidades

En las dos muestras individuales examinadas, **{mayor_confianza}** presentó el valor máximo de Softmax mayor. En MNIST, el máximo fue **{m['confianza']:.4%}**; en Fashion-MNIST fue **{f['confianza']:.4%}**. Los dos vectores son distribuciones de 10 clases, no porcentajes de precisión global. Un Softmax dominante puede ocurrir incluso en una predicción equivocada: **confianza alta no garantiza acierto**. Tampoco permite concluir una diferencia general de calibración usando solo dos imágenes. Para evaluar confianza de forma más rigurosa habría que considerar un conjunto grande de predicciones, sus etiquetas reales y métricas de calibración.

### Evidencia visual directa de los datos

![Primeras nueve imágenes MNIST](evidencias/mnist_muestras.png)

![Primeras nueve imágenes Fashion-MNIST](evidencias/fashion_muestras.png)

**Observación:** las imágenes MNIST muestran trazos de escritura manuscrita aislados sobre fondo oscuro; las de Fashion-MNIST muestran siluetas completas de prendas, con zonas sombreadas y rasgos que pueden solaparse entre categorías. El cambio de dominio aumenta la dificultad para separar ciertas clases visualmente similares. Debe evitarse atribuir un nivel fijo de confianza al dominio completo a partir de un único ejemplo.
"""


def main():
    m = abrir_resultado("mnist")
    f = abrir_resultado("fashion")
    if not README.exists():
        raise SystemExit("No existe el README.md: coloca este archivo en la raíz del proyecto.")
    fuente = README.read_text(encoding="utf-8")
    if fuente.count(MARCADOR_INICIO) != 1 or fuente.count(MARCADOR_FIN) != 1:
        raise SystemExit("El README debe contener exactamente un par de marcadores para los resultados.")

    crear_grafica(
        m, f, "accuracy", "Accuracy después de 1 época con 5000 imágenes",
        "comparacion_accuracy.png", "Accuracy (proporción)",
    )
    crear_grafica(
        m, f, "loss", "Loss después de 1 época con 5000 imágenes",
        "comparacion_loss.png", "Loss (entropía cruzada)",
    )
    antes, resto = fuente.split(MARCADOR_INICIO, 1)
    _, despues = resto.split(MARCADOR_FIN, 1)
    bloque = bloque_resultados(m, f)
    README.write_text(
        antes + MARCADOR_INICIO + "\n" + bloque + "\n" + MARCADOR_FIN + despues,
        encoding="utf-8",
    )
    print("\nCOMPARACIÓN GENERADA CON RESULTADOS REALES")
    print(f"MNIST       : train accuracy={m['train_accuracy']:.4%}; train loss={m['train_loss']:.5f}")
    print(f"Fashion-MNIST: train accuracy={f['train_accuracy']:.4%}; train loss={f['train_loss']:.5f}")
    print("Archivos actualizados:")
    print("  README.md (sección automática: resultados y análisis)")
    print("  evidencias/comparacion_accuracy.png")
    print("  evidencias/comparacion_loss.png")
    print("Todo listo para revisar el README y subirlo a GitHub.")


if __name__ == "__main__":
    main()
