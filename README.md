# Tarea 2 – Ciencia de Datos

Este repositorio contiene el desarrollo de la **Tarea 2** de la materia *Introducción a la Ciencia de Datos*.  
El trabajo está dividido en dos partes complementarias:

---

## Parte 1 – Análisis de datos reales (Bank Marketing)

En esta parte se analiza un conjunto de datos de campañas de *marketing directo* de una institución bancaria portuguesa (mayo 2008 – noviembre 2010).  

- **Fuente:** Conjunto de datos UCI *Bank Marketing*, elaborado por Paulo Cortez (Universidad de Minho), Sérgio Moro y Paulo Rita (ISCTE-IUL) en 2014.  
- **Objetivo:** promover un *depósito a plazo fijo* mediante campañas de llamadas telefónicas.  
- **Variable respuesta:** binaria (`yes` / `no`), indicando si el cliente suscribió el depósito.  

Se realizaron las siguientes tareas:
- Limpieza y codificación de los datos.  
- Análisis descriptivo de variables y problemáticas del dataset.  
- Implementación de modelos de clasificación:
  - Naive Bayes  
  - LDA (Linear Discriminant Analysis)  
  - QDA (Quadratic Discriminant Analysis)  
  - Proyección de Fisher  
  - k-NN (k–Nearest Neighbors)  
  - Regresión Logística  

El reporte completo se encuentra en:  
📄 `Parte1/Tarea_2__Parte_1__Introduccion_a_la_...pdf`

---

## Parte 2 – Estudio de simulación con clasificadores

En esta parte se construyó un estudio de simulación tipo **Monte Carlo** para comparar distintos clasificadores bajo escenarios controlados.

- **Clasificadores considerados:**
  - Bayes (óptimo teórico, usado como referencia)  
  - LDA  
  - QDA  
  - k-NN  
  - Proyección de Fisher  

- **Escenarios simulados:**
  1. Covarianzas iguales (LDA óptimo)  
  2. Covarianzas distintas (QDA óptimo)  
  3. Desbalance de clases  
  4. Alta correlación  
  5. Medias cercanas (escenario difícil con traslape)  

- **Diseño experimental:**
  - Tamaños muestrales: `n ∈ {50, 100, 200, 500}`  
  - Vecinos de k-NN: `k ∈ {1, 3, 5, 11, 21}`  
  - Réplicas: `R = 20` corridas independientes para promediar error y desviación estándar  

- **Productos generados:**
  - Gráficas de evolución del riesgo por método  
  - Comparaciones entre riesgo estimado por validación cruzada y riesgo verdadero de Bayes  
  - Tablas de resumen de desempeño por escenario  

El código se encuentra en:  
 `Parte2/compare_classfiers/`  
- `_functions.py`: generación de datos y lógica de simulación  
- `_plots.py`: funciones de graficación  
- `main.py`: script principal de ejecución  
- `Run_program.ipynb`: notebook demostrativo  

Reporte asociado:  
 `Parte2/Tarea_2__Parte_2__Introduccion_a_la_...pdf`

---

## ▶ Ejecución

Para correr la **Parte 2** desde terminal:

```bash
cd Parte2
python main.py
