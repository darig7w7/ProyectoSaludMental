# ProyectoSaludMental
Proyecto de Ciencia de Datos usando el dataset OSMI Mental Health in Tech Survey para predecir la variable treatment (búsqueda de tratamiento en salud mental).

# Salud Mental en el Trabajo Tecnológico
### Análisis predictivo sobre búsqueda de tratamiento psicológico en profesionales de TI

---

## Descripción

Este proyecto analiza datos de una encuesta aplicada a profesionales del sector tecnológico con el objetivo de identificar los factores del entorno laboral que se asocian con la decisión de buscar tratamiento de salud mental. Se implementaron técnicas de limpieza de datos, visualización exploratoria y clasificación supervisada para construir un modelo predictivo interpretable.

El dataset utilizado es el **OSMI Mental Health in Tech Survey**, disponible públicamente en Kaggle.

---

## Problema de investigación

¿Qué características del entorno laboral (tipo de empresa, beneficios, cultura organizacional) permiten predecir si un trabajador de tecnología buscará tratamiento para un problema de salud mental?

---

## Dataset

| Característica | Detalle |
|---|---|
| Fuente | OSMI Mental Health in Tech Survey |
| Registros | 1,259 respuestas válidas (tras limpieza) |
| Variables originales | 27 columnas |
| Variables usadas en modelo | 23 columnas |
| Variable objetivo | `treatment` (buscó tratamiento: Sí / No) |

La muestra está compuesta principalmente por hombres (78.9%), con edades concentradas entre los 25 y 40 años y una mediana de 31 años.

---

## Estructura del proyecto

```
├── survey.csv                  # Dataset original
├── analisis_salud_mental.ipynb # Notebook principal
└── README.md
```

---

## Pipeline de análisis

### 1. Carga y exploración inicial
Lectura del CSV desde Google Drive. Primera inspección con `df.describe()` y `df.head()`. Se detectaron valores extremos en la columna `Age` (negativos y mayores a 100), así como alta variabilidad de texto libre en `Gender`.

### 2. Limpieza de datos

**Edad:** Se calculó la mediana de edades válidas (entre 15 y 80 años) y se usó para reemplazar valores fuera de ese rango.

```python
mediana_edad = df.loc[(df['Age'] >= 15) & (df['Age'] <= 80), 'Age'].median()
df['Age'] = df['Age'].apply(lambda x: mediana_edad if not (15 <= x <= 80) else x)
```

**Género:** Se estandarizaron manualmente más de 40 variantes textuales a tres categorías (`Male`, `Female`, `Other`) mediante una función de mapeo. Se aplicó adicionalmente `IsolationForest` con `TfidfVectorizer` a nivel de caracteres para detectar anomalías residuales — resultado: 0 anomalías encontradas tras la estandarización.

**Nulos:**
- `work_interfere` → imputado con `'Unknown'`
- `self_employed` → imputado con `'No'`
- `comments`, `state`, `Country`, `Timestamp` → eliminados (irrelevantes para el modelo o con alta cardinalidad/nulidad)

### 3. Codificación

Todas las variables categóricas fueron codificadas con `LabelEncoder`, guardando cada encoder en un diccionario para permitir inversión posterior.

### 4. Escalado y división

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Se usó `stratify=y` dado que las clases en `treatment` están levemente desbalanceadas (~50.6% / 49.4%).

### 5. Modelos entrenados

| Modelo | Accuracy |
|---|---|
| Regresión Logística | 68.3% |
| **Árbol de Decisión** | **82.1%** |

El Árbol de Decisión (`max_depth=5`) superó a la Regresión Logística en 13.8 puntos porcentuales.

### 6. Evaluación del mejor modelo (Árbol de Decisión)

```
                precision    recall    f1-score   support
No tratamiento     0.92      0.69       0.79       124
Con tratamiento    0.76      0.95       0.84       128

      accuracy                          0.82       252
     macro avg     0.84      0.82       0.82       252
  weighted avg     0.84      0.82       0.82       252
```

El modelo tiene un recall alto (0.95) para la clase "Con tratamiento", lo que lo hace robusto para identificar a trabajadores que sí buscarían ayuda — comportamiento deseable en un contexto de salud pública.

---

## Hallazgos principales

- La distribución de la variable objetivo está casi balanceada, lo que facilita el entrenamiento sin técnicas de sobremuestreo.
- Variables como `family_history`, `benefits` y `work_interfere` mostraron alta frecuencia de valores que intuitivamente se asocian con la búsqueda de tratamiento.
- El árbol de decisión alcanzó mayor capacidad predictiva que la regresión logística, posiblemente por la naturaleza no lineal de las interacciones entre variables del entorno laboral.
- La limpieza de la columna `Gender` fue el paso más crítico del preprocesamiento, dado el alto ruido textual presente en las respuestas originales.

---

## Tecnologías utilizadas

- Python 3.x
- pandas / numpy
- matplotlib / seaborn
- scikit-learn (`TfidfVectorizer`, `IsolationForest`, `LabelEncoder`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`)
- Google Colab

---

## Cómo reproducir el análisis

1. Clonar el repositorio
2. Subir `survey.csv` a Google Drive en la ruta `/MyDrive/Ciencia de datos/Programación Ciencia de Datos/`
3. Abrir `analisis_salud_mental.ipynb` en Google Colab
4. Ejecutar todas las celdas en orden

---

## Limitaciones

- El dataset proviene de una encuesta voluntaria, por lo que puede existir sesgo de autoselección.
- La muestra está concentrada en hombres jóvenes de habla inglesa, lo que limita la generalización a otros contextos.
- Se eliminaron variables potencialmente útiles como `Country` y `state` por alta cardinalidad o porcentaje de nulos.

---

## Autores

Trabajo desarrollado en el marco del curso **Programación en Ciencia de Datos**.

