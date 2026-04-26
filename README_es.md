# IoT Predictive Maintenance at the Edge (Raspberry Pi + Machine Learning)

> 🇺🇸 **English version available:** [Click here for the English README](README.md)

# Mantenimiento Predictivo IoT en el Edge (Raspberry Pi + Machine Learning)

![Arquitectura o foto del hardware](/images/Full_setup.jpg)

Este proyecto es un Producto Mínimo Viable (MVP) industrial diseñado para detectar fallos mecánicos de forma anticipada en maquinaria rotativa. Utiliza procesamiento de señales en el *Edge* y aprendizaje no supervisado (**Isolation Forest**) para evitar paradas imprevistas (lucro cesante) y mantenimientos preventivos innecesarios.

## Arquitectura del Sistema

El sistema captura vibraciones, las procesa localmente y alerta en tiempo real si la máquina sale de su "burbuja de normalidad".

* **Hardware (Edge):** Raspberry Pi 2 + Acelerómetro MEMS triaxial (MPU-6050) conectado vía I2C.
* **Procesamiento:** Extracción de variables en el dominio del tiempo (RMS, Curtosis) y de frecuencia (Transformada Rápida de Fourier - FFT) mediante Python.
* **Almacenamiento y Visualización:** InfluxDB local (Time-Series DB) + Cuadro de mando en Grafana.
* **Machine Learning:** `IsolationForest` de Scikit-Learn entrenado exclusivamente con el estado "sano" de la máquina.
* **Sistema de Alertas:** Evaluación mediante "Ventanas Deslizantes" de tiempo y envío de alarmas automatizadas vía API de Telegram.

## Resultados y Validación

El núcleo matemático fue validado inicialmente con el dataset industrial de la *Case Western Reserve University (CWRU)*, logrando un **91% de precisión y un 0% de Falsas Alarmas**.

Para la validación física, se inyectaron fallos mecánicos reales en un motor rotativo (ventilador). 

![Dashboard Grafana y Alerta Telegram](/images/Grafana_Pannel.jpeg)

**Logro clave:** Tras implementar una lógica de persistencia de fallos (evaluación por ventanas de 1 minuto para ignorar ruidos aislados), el sistema alcanzó un **100% de detección de anomalías**.

> **Documentación Completa:** Para un desglose técnico detallado del código, la justificación de negocio y el reporte exhaustivo de las pruebas físicas, consulta la carpeta `/docs` de este repositorio.

## Cómo ejecutar este proyecto

1. **Montaje del Hardware:** Conecta el sensor MPU-6050 a la Raspberry Pi utilizando los pines del bus I2C (SDA al pin 3, SCL al pin 5, VCC a 3.3V y GND a GND). Fija el acelerómetro rígidamente a la carcasa del motor que deseas monitorizar para asegurar una transmisión pura de la vibración.
2. **Configuración del Entorno (Software):** Instala InfluxDB (para almacenar series temporales) y Grafana (para visualizar el cuadro de mando) en tu Raspberry Pi. Habla con el *BotFather* en Telegram para crear un Bot y obtener tu Token de acceso.
3. **Captura de la Línea Base (Dataset):** Enciende la máquina y asegúrate de que funciona en estado óptimo (sano). Ejecuta el script `1_generar_dataset.py` para que el sistema empiece a registrar las vibraciones y características físicas, creando tu dataset de normalidad.
4. **Entrenamiento del Modelo:** Una vez tengas suficientes datos sanos, ejecuta el script `2_entrenar_modelo.py`. El algoritmo de aprendizaje no supervisado creará su "burbuja de normalidad" y guardará el modelo entrenado en la memoria local.
5. **Evaluación del Modelo (Testing):** Antes de desplegar el sistema, ejecuta `3_evaluar_modelo.py`. Este script pondrá a prueba el modelo generado comprobando su precisión, la matriz de confusión y verificando que el índice de falsas alarmas sea bajo para garantizar su viabilidad comercial.
6. **Puesta en Producción (Monitorización):** Ejecuta el script principal `4_monitorizar_en_produccion.py`. El sistema pasará a modo de escucha activa, evaluando ventanas deslizantes de tiempo. Si la vibración se sale de la burbuja geométrica de forma persistente, el bot de Telegram disparará la alerta a tu dispositivo móvil.

## Futuras Mejoras Planteadas
* Transición a **Motor Current Signature Analysis (MCSA)** para una lectura 100% no invasiva midiendo el amperaje en el cuadro eléctrico.
* Implementación de **Order Tracking** (variables adimensionales) para adaptar la IA a maquinaria de velocidad variable.
