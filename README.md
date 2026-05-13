# Análisis de salud mental en la industria tecnológica mediante aprendizaje automático

**Darwin Rigoberto Mamani Quispe · Melbis Chino Huaycani**  
`darig.mq@gmail.com` · `melbishuaycani@gmail.com`

---

## Resumen

Se analiza el dataset *OSMI Mental Health in Tech Survey 2014* con técnicas de aprendizaje automático para predecir si un profesional del sector tecnológico busca tratamiento de salud mental. El pipeline incluye detección de anomalías en texto con TF-IDF e Isolation Forest, imputación diferenciada de valores nulos y comparación entre Regresión Logística y Árbol de Decisión. El Árbol de Decisión alcanzó **82.1 % de accuracy** con un recall de **0.95** para la clase positiva. El análisis de importancia de variables reveló que `work_interfere` concentra el 82.8 % del poder predictivo, lo que plantea consideraciones sobre *data leakage* parcial en aplicaciones reales.

---

## Contexto

La salud mental en la industria tecnológica presenta una prevalencia significativa de trastornos como depresión y ansiedad, agravada por culturas organizacionales que históricamente han valorado la resiliencia sobre la vulnerabilidad. La iniciativa **Open Sourcing Mental Illness (OSMI)** recopila desde 2014 datos longitudinales sobre actitudes y comportamientos en este sector.

Este trabajo tiene dos objetivos:
1. Identificar los factores laborales y demográficos más asociados a la búsqueda de tratamiento.
2. Construir modelos predictivos que estimen esa decisión a partir del contexto laboral del trabajador.

---

## Dataset

| Característica | Detalle |
|---|---|
| Fuente | [OSMI Mental Health in Tech Survey 2014](https://osmihelp.org/research) |
| Registros (tras limpieza) | 1,259 |
| Variables originales | 27 |
| Variables utilizadas en modelo | 23 |
| Variable objetivo | `treatment` — buscó tratamiento (Sí / No) |
| Balance de clases | 50.8 % positivo · 49.2 % negativo |
| Países representados | 48 |

**Variables principales:**

| Variable | Tipo | Descripción |
|---|---|---|
| `treatment` | Binaria | Variable objetivo |
| `work_interfere` | Ordinal | Interferencia de la condición con el trabajo |
| `family_history` | Binaria | Historial familiar de enfermedad mental |
| `benefits` | Nominal | Beneficios de salud mental del empleador |
| `anonymity` | Nominal | Protección del anonimato |
| `Age` | Numérica | Edad del encuestado |
| `Gender` | Nominal | Género (3 categorías tras normalización) |
| `no_employees` | Ordinal | Tamaño de la empresa |

---

## Pipeline

### 1. Limpieza de edad

El campo `Age` contenía valores fuera de rango producto de entrada libre de texto (mínimo: −1726, máximo: 1×10¹¹). Se calculó la mediana sobre el rango válido [15, 80] y se usó como valor de reemplazo.

```python
mediana_edad = df.loc[(df['Age'] >= 15) & (df['Age'] <= 80), 'Age'].median()
df['Age'] = df['Age'].apply(lambda x: mediana_edad if not (15 <= x <= 80) else x)
df['Age'] = df['Age'].astype(int)
```

### 2. Normalización de género

La columna presentaba más de 40 variantes de texto libre. Se implementó una función de mapeo explícito a tres categorías (`Male`, `Female`, `Other`), seguida de una auditoría con `TfidfVectorizer` (nivel de caracteres, n-gramas 2–3) e `IsolationForest`. Resultado: **0 anomalías** detectadas tras la estandarización.

```python
def estandarizar_genero(g):
    g_clean = str(g).lower().strip()
    if g_clean in ['female', 'f', 'woman', 'femail', 'cis female', ...]:
        return 'Female'
    if g_clean in ['male', 'm', 'man', 'make', 'cis male', ...]:
        return 'Male'
    return 'Other'
```

Distribución resultante: **78.9 % Male** (n=993) · **19.8 % Female** (n=249) · **1.3 % Other** (n=17).

### 3. Tratamiento de valores nulos

| Columna | Nulos | % | Estrategia |
|---|---|---|---|
| `work_interfere` | 264 | 21 % | Categoría `'Unknown'` |
| `self_employed` | 18 | 1.4 % | Moda (`'No'`) |
| `state` | 515 | 41 % | Eliminada |
| `comments` | 1,095 | 87 % | Eliminada |
| `Country` | — | — | Eliminada (48 valores únicos, alta cardinalidad) |
| `Timestamp` | — | — | Eliminada (no predictiva) |

### 4. Codificación y escalado

Todas las variables categóricas fueron codificadas con `LabelEncoder`, guardando cada encoder en un diccionario para permitir inversión posterior. Se aplicó `StandardScaler` ajustado únicamente sobre el conjunto de entrenamiento.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Se usó `stratify=y` dado el leve desbalance entre clases (~50.6 % / 49.4 %).

---

## Modelos

```python
# Regresión Logística
m_log = LogisticRegression(max_iter=1000)
m_log.fit(X_train_scaled, y_train)

# Árbol de Decisión
m_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
m_tree.fit(X_train, y_train)
```

### Resultados comparativos (n=252, conjunto de prueba)

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| Regresión Logística | 0.683 | 0.74 | 0.68 | 0.70 |
| **Árbol de Decisión** | **0.821** | **0.84** | **0.82** | **0.82** |
| Árbol sin `work_interfere` | 0.623 | 0.65 | 0.62 | 0.63 |

El Árbol de Decisión identificó correctamente el **95 % de los casos positivos reales** (solo 6 falsos negativos de 128 casos positivos totales).

### Importancia de variables — Árbol de Decisión

| Variable | Importancia | Acumulado |
|---|---|---|
| `work_interfere`   | 0.828 | 82.8 % |
| `family_history` | 0.054 | 88.3 % |
| `benefits` | 0.040 | 92.3 % |
| `anonymity` | 0.023 | 94.6 % |
| `care_options` | 0.015 | 96.1 % |
| Resto (17 variables) | 0.039 | 100 % |

### Data leakage parcial en `work_interfere`

La pregunta que origina esta variable presupone la existencia de una condición de salud mental, introduciendo un **target leakage parcial**. El modelo que la excluye cae 19.8 puntos porcentuales de accuracy (82.1 % → 62.3 %), revelando que sin ella los predictores estructuralmente más sólidos son `benefits` y `anonymity` — variables directamente relacionadas con las políticas organizacionales del empleador.

> El modelo con `work_interfere` es eficaz para diagnóstico de situaciones actuales. El modelo sin ella es más adecuado para estudios de prevención temprana.

---

## Conclusiones

1. Un pipeline de preprocesamiento riguroso (detección de anomalías, imputación diferenciada, normalización de texto libre) es determinante para la calidad del modelo.
2. El Árbol de Decisión (`max_depth=5`) superó a la Regresión Logística en todas las métricas, con un margen de 13.8 puntos porcentuales de accuracy.
3. El historial familiar y las políticas de salud mental del empleador son predictores significativos **independientemente** de `work_interfere`.
4. La identificación y cuantificación del data leakage (+19.8 pp) es una contribución metodológica concreta para futuros trabajos con este dataset.

---

## Trabajo futuro

- **Análisis longitudinal 2014–2016:** evaluar cambios en actitudes y patrones predictivos con el dataset OSMI 2016.
- **World Happiness Report 2016:** determinar si el contexto sociocultural nacional modera el efecto de las políticas organizacionales.
- **Modelos de mayor capacidad:** Random Forest, XGBoost y ensambles.
- **Análisis causal:** DAGs y modelos de ecuaciones estructurales para distinguir predictores de variables confusoras.

---

## Reproducción

1. Clonar el repositorio
2. Colocar `survey.csv` en `/content/drive/MyDrive/Ciencia de datos/Programación Ciencia de Datos/`
3. Abrir `analisis_salud_mental.ipynb` en Google Colab
4. Ejecutar todas las celdas en orden

**Dependencias:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

---

## Referencias

1. K. Goodman et al., "Workplace mental health in the technology sector: A systematic review," *J. Occupational Health Psychol.*, vol. 24, no. 3, pp. 312–328, 2019.
2. OSMI, "Mental Health in Tech Survey 2014," Open Sourcing Mental Illness. [En línea]: https://osmihelp.org/research
3. World Health Organization, "Mental Health: Strengthening Our Response," WHO Fact Sheet, 2022.
4. T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.
5. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
