# Guía paso a paso: Visual Studio Code + GitHub

## ETAPA 1. Abrir el proyecto

1. Descomprime `practica-cnn-comparativa.zip` en una carpeta conocida, por ejemplo **Documentos**.
2. Inicia **Visual Studio Code**.
3. Pulsa **Archivo > Abrir carpeta...** y selecciona **practica-cnn-comparativa** (no la carpeta superior).
4. Instala la extensión oficial **Python** de Microsoft desde **Extensiones** si todavía no está instalada.
5. Abre **Terminal > Nuevo terminal**. Verifica que la ruta mostrada termina en `practica-cnn-comparativa`.

## ETAPA 2. Configurar Python (PowerShell de Windows)

Recomendado: Python **3.11** (64 bits), que es compatible con las dependencias indicadas. Verifica la instalación:

```powershell
py -0p
```

Si Python 3.11 aparece en la lista, crea el entorno dentro de la carpeta:

```powershell
py -3.11 -m venv .venv
```

Instala los paquetes (no necesitas activar el entorno, así evitas problemas de políticas de PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Confirma que TensorFlow se importó correctamente:

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf; import matplotlib; print('TensorFlow:', tf.__version__); print('Matplotlib:', matplotlib.__version__)"
```

**Opcional pero recomendable:** `Ctrl+Shift+P > Python: Select Interpreter > Enter interpreter path` y selecciona el archivo `.venv\Scripts\python.exe`. De esa forma el botón de ejecución de VS Code apunta al mismo Python que se usó para instalar TensorFlow.

## ETAPA 3. Ejecutar la PARTE 1 (MNIST)

```powershell
.\.venv\Scripts\python.exe mnist_cnn.py
```

1. Aparece la cuadrícula de **9 dígitos** de MNIST. Examina trazos y etiquetas; se guarda `evidencias/mnist_muestras.png`.
2. **Cierra la ventana de la cuadrícula**, porque Matplotlib puede esperar a que se cierre antes de entrenar.
3. Verifica que `model.summary()` muestre **121,930 parámetros**.
4. Espera a que se complete la **época 1/1** con 5,000 muestras.
5. Se muestra pérdida y precisión de entrenamiento y de prueba.
6. Comprueba que se impriman **10 probabilidades Softmax**, la suma (próxima a 1), el índice elegido y la etiqueta real.
7. Aparece la gráfica de la primera muestra de prueba. Ciérrala para que concluya el programa.
8. Revisa `resultados/mnist_resultados.json` y `evidencias/mnist_prediccion.png`.

## ETAPA 4. Ejecutar la PARTE 2 (Fashion-MNIST)

```powershell
.\.venv\Scripts\python.exe fashion_cnn.py
```

Repite el procedimiento. En este caso aparecen **9 prendas**, la capa densa es de 128 neuronas, el resumen debe arrojar **225,034 parámetros** y las clases predichas se muestran con su nombre legible. Se crean `evidencias/fashion_muestras.png`, `evidencias/fashion_prediccion.png` y `resultados/fashion_resultados.json`.

## ETAPA 5. Generar el análisis real para la PARTE 3

```powershell
.\.venv\Scripts\python.exe generar_comparacion.py
```

El programa toma los dos archivos JSON anteriores y crea `evidencias/comparacion_accuracy.png` y `evidencias/comparacion_loss.png`. También actualiza **únicamente** la sección 8 del `README.md` con:

- Tabla de métricas reales de entrenamiento y prueba.
- Los diez valores Softmax de cada imagen, la suma y `np.argmax`.
- Interpretación de clase predicha vs. clase real y acierto/error.
- Comparación de confianza y una discusión de límites metodológicos.
- Las seis figuras de evidencia enlazadas para verse directamente en GitHub.

**Importante:** revisa el README completo usando `Ctrl+Shift+V` (vista previa de Markdown). Asegúrate de que se ven las seis imágenes antes de entregar.

## ETAPA 6. Preparar GitHub con un historial ordenado

Crea en github.com un repositorio **público** llamado `practica-cnn-comparativa`, **vacío** (sin README ni `.gitignore` inicial). Para cumplir el criterio de historial de commits, realiza los siguientes pasos **en este orden**:

**Primer commit: código y configuración**

```powershell
git init
git add mnist_cnn.py fashion_cnn.py requirements.txt .gitignore
git commit -m "Implementar redes CNN de MNIST y Fashion-MNIST"
```

**Segundo commit: documentación y comparación**

```powershell
git add README.md generar_comparacion.py INSTRUCCIONES_VSCODE.md
git commit -m "Documentar arquitectura y analisis comparativo de CNN"
```

**Tercer commit: resultados reales y figuras** (tras ejecutar la práctica):

```powershell
git add README.md evidencias/ resultados/
git commit -m "Agregar evidencias y resultados reales de ambos experimentos"
```

Si ya habías ejecutado los scripts antes de hacer los primeros commits, puedes igualmente seguir estas etapas de `git add` por grupos, pero no repitas comandos que creen commits vacíos. Comprueba el historial:

```powershell
git log --oneline
```

Vincula el repositorio cambiando `TU_USUARIO` por tu usuario verdadero de GitHub:

```powershell
git branch -M main
git remote add origin https://github.com/TU_USUARIO/practica-cnn-comparativa.git
git push -u origin main
```

Si pide configurar identidad al hacer commit, usa `git config --global user.name "Tu nombre de Git"` y `git config --global user.email "tu-correo-de-git"`; no necesitas compartir esos datos aquí. La autenticación para el push puede abrir el navegador.

Finalmente, abre en un navegador la dirección del repositorio y comprueba que sea **público** y que las imágenes se vean en la página del README.

## Lista final de verificación según la rúbrica

- [ ] Ambos scripts corren de inicio a fin sin errores.
- [ ] Se abren las visualizaciones de 9 imágenes y se guardan sus PNG.
- [ ] `model.summary()` muestra las capas y parámetros exactos de la guía.
- [ ] Cada entrenamiento utiliza 5,000 muestras y una época.
- [ ] Se imprime e interpreta el vector Softmax y su suma.
- [ ] Fashion-MNIST transforma índices enteros usando `class_names`.
- [ ] README contiene todas las respuestas de las partes 1, 2 y 3.
- [ ] README incluye la tabla real y todas las figuras.
- [ ] Git tiene commits significativos, repositorio público y enlace funcional.

## Errores frecuentes

- **`py` no se reconoce:** Python no está instalado correctamente; instala Python 3.11 y reinicia VS Code.
- **`Requested Python version not installed`:** revisa `py -0p`, instala Python 3.11 o selecciona una versión instalada compatible.
- **`No module named tensorflow`:** confirma que los paquetes se instalaron usando el Python de `.venv` y que VS Code usa ese intérprete.
- **No comienza el entrenamiento:** es posible que una ventana de Matplotlib siga abierta; ciérrala.
- **La descarga del dataset falla:** se necesita internet al menos en la primera descarga a la caché local de Keras.
- **`generar_comparacion.py` indica que falta JSON:** primero termina la ejecución de ambos scripts.
- **Git rechaza un push:** verifica URL remota, permisos y autenticación; evita generar un README desde la interfaz al crear un repositorio nuevo.
