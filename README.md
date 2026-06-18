# dl_rl_buck

Identificación de sistemas y control de voltaje mediante aprendizaje por refuerzo para un convertidor Buck DC-DC, utilizando Redes de Estado de Eco (ESN) y el algoritmo Soft Actor-Critic (SAC).

El objetivo principal era reemplazar un controlador PID que exhibía una oscilación persistente de 50-200 Hz por un controlador aprendido, entrenado completamente a partir de datos experimentales de una planta de hardware real (ESP32-S3 + convertidor Buck). El proyecto se encuentra documentado en `dl_buck-1.pdf`.

## Estructura del Repositorio


```

```text
File generated successfully.


```

dl_rl_buck/
|-- bode_analysis.py        Estimación de la respuesta en frecuencia a partir de capturas de chirp
|-- data_preparation.py     Calibración de ADC, verificación de datos y generación de manifiesto
|-- train_plants.py         Entrenamiento de modelos ESN y líneas base (LSTM/GRU/NARX-MLP)
|-- preflight_checks.py     Verificaciones de cordura para modelos de plantas entrenados antes de su uso en RL
|-- rl_pipeline.py          Entorno Gym que envuelve la planta para el entrenamiento de RL
|-- misc/
|   |-- buck_host.py        Interfaz serie en el lado del host para la captura de datos del ESP32-S3
|   |-- esp32_s3_fw.7z      Firmware del ESP32-S3 (Firmware de identificación del convertidor Buck V2)
|-- datos/
|   |-- data_buck1-4.7z     Capturas experimentales crudas de la planta de hardware
|-- res_and_img/            Gráficos de resultados y figuras utilizadas en el informe
|-- dl_buck-1.pdf           Informe del proyecto

```

## Hardware

- **Microcontrolador:** ESP32-S3
- **Planta:** Convertidor Buck DC-DC
- **ADC:** 80 kHz, 12 bits
- **Tasa del lazo de control:** 2 kHz
- **Frecuencia PWM:** ~20 kHz
- **Rango de voltaje de entrada probado:** 9-31 V
- **Rango de resistencia de carga probado:** 7.2-16.27 ohmios

## Dependencias


```

numpy
scipy
matplotlib
stable-baselines3
gymnasium
tensorflow   # opcional, requerido únicamente para las líneas base LSTM/GRU/NARX-MLP
pyserial     # requerido únicamente para buck_host.py

```

Instalar con:


```

pip install numpy scipy matplotlib stable-baselines3 gymnasium pyserial

```

TensorFlow es opcional. Si no está instalado, `train_plants.py` omitirá las líneas base de aprendizaje profundo y entrenará únicamente el modelo ESN.

---

## Uso

### 1. Captura de datos (requiere hardware)

Conecte el ESP32-S3 y use `buck_host.py` para capturar datos:


```

# Sesión interactiva

python misc/buck_host.py /dev/ttyUSB0

# Captura de secuencia en un solo disparo (one-shot)

python misc/buck_host.py /dev/ttyUSB0 --seq -o mi_captura.npz

# Captura de chirp (ciclo de trabajo nominal 50%, amplitud 5%, 100-8000 Hz, 4 segundos)

python misc/buck_host.py /dev/ttyUSB0 --chirp 50 5 100 8000 4 -o chirp1.npz

# Captura de retención (hold) (30 segundos, ciclo de trabajo fijo)

python misc/buck_host.py /dev/ttyUSB0 --hold 30 -o hold1.npz

```

Las capturas se guardan como archivos `.npz` autodescriptivos que contienen las muestras crudas del ADC, el ciclo de trabajo reconstruido por muestra, la tasa de muestreo y los metadatos de la captura.

### 2. Preparación de datos

Verifique y compile las capturas crudas en un conjunto de datos limpio:


```

python data_preparation.py <carpeta_cruda1> <carpeta_cruda2> ...

```

Aplica una calibración de ADC a voltaje utilizando un interpolador Pchip monotónico, realiza comprobaciones de integridad y escribe un índice en `manifest.csv`.

### 3. Análisis de la respuesta en frecuencia


```

python bode_analysis.py <captura_chirp.npz>

```

Calcula la función de respuesta en frecuencia (FRF) a partir de una captura de chirp utilizando una ventana rectangular (correcta para chirps lineales) y guarda un gráfico comparativo frente a la ventana de Hann como referencia.

### 4. Entrenar modelos de la planta


```

# Entrenar solo ESN

python train_plants.py datos/compiled_data --mode reservoir

# Entrenar solo líneas base LSTM/GRU/NARX-MLP (requiere TensorFlow)

python train_plants.py datos/compiled_data --mode baseline

# Entrenar todo

python train_plants.py datos/compiled_data --mode all

```

Genera un modelo ESN entrenado guardado como `esn_compiled_model.npz`.

### 5. Verificaciones previas al vuelo (Preflight checks)

Verifique un modelo entrenado antes de usarlo en aprendizaje por refuerzo (RL):


```

python preflight_checks.py esn_compiled_model.npz

```

Ejecuta comprobaciones de determinismo, monotonicidad de la ganancia de DC y estabilidad.

### 6. Entrenamiento de RL

`rl_pipeline.py` define el entorno de Gym `BuckEnv`. Entrene utilizando stable-baselines3:

```python
from stable_baselines3 import SAC
from rl_pipeline import BuckEnv
from plants import EchoStateNetworkPlant

plant = EchoStateNetworkPlant("esn_compiled_model.npz")
env = BuckEnv(plant)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

```

## Resumen de Resultados

La arquitectura multiescala ESN v4 (350 neuronas rápidas con un ancho de banda de 2546 Hz + 150 neuronas lentas con 167 ms de memoria) logró un RMSE de despliegue (rollout) en lazo abierto de 284.7 mV, superando a las líneas base GRU, LSTM y ESN de escala única.

El entrenamiento de RL en lazo cerrado no alcanzó el objetivo de regulación de un error inferior a 100 mV en todas las condiciones de operación. Los factores contribuyentes conocidos están documentados en el informe y en el registro de experimentos; incluyen la inestabilidad de arranque en frío (cold-start) del reservorio, un posible cambio en la distribución (distribution shift) entre los datos de entrenamiento en lazo abierto y las acciones del agente en lazo cerrado, y una probable operación en modo de conducción discontinua (DCM) bajo condiciones extremas de Vin y carga. Ninguno de estos factores ha sido verificado mediante experimentos controlados.

## Notas

* Los datos de fase producidos por `bode_analysis.py` en las capturas del hardware real no son fiables debido a la acumulación de fase dominada por el ruido por debajo de la banda del chirp. Los datos de ganancia sí son utilizables.
* El modelo ESN requiere un período de calentamiento (warm-up) de aproximadamente 835 ms (cinco constantes de tiempo de las neuronas lentas) para ofrecer predicciones fiables. Su uso sin este calentamiento produce salidas incorrectas con ciclos de trabajo bajos.
* Las condiciones de operación probablemente en DCM (Vin por debajo de 15 V o por encima de 20 V con R = 16.27 ohmios) presentaron un RMSE superior a 1700 mV y fueron excluidas de las condiciones de entrenamiento de RL.
