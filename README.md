# IoT Predictive Maintenance at the Edge (Raspberry Pi + Machine Learning)

> 🇪🇸 **Versión en español disponible:** [Haz clic aquí para leer el README en español](README_es.md)

![Architecture or hardware photo](/images/Full_setup.jpeg)

This project is an industrial Minimum Viable Product (MVP) designed for the early detection of mechanical failures in rotating machinery. It uses *Edge* signal processing and unsupervised learning (**Isolation Forest**) to prevent unplanned downtime (loss of revenue) and unnecessary preventive maintenance.

## System Architecture

The system captures vibrations, processes them locally, and sends real-time alerts if the machine deviates from its "bubble of normality".

* **Hardware (Edge):** Raspberry Pi 2 + Triaxial MEMS Accelerometer (MPU-6050) connected via I2C.
* **Processing:** Feature extraction in the time domain (RMS, Kurtosis) and frequency domain (Fast Fourier Transform - FFT) using Python.
* **Storage and Visualization:** Local InfluxDB (Time-Series DB) + Grafana Dashboard.
* **Machine Learning:** Scikit-Learn's `IsolationForest` trained exclusively on the "healthy" state of the machine.
* **Alert System:** Evaluation using time-based "Sliding Windows" and automated alarm generation via the Telegram API.

## Results and Validation

The mathematical core was initially validated using the industrial dataset from *Case Western Reserve University (CWRU)*, achieving **91% accuracy and a 0% False Alarm rate**.

For physical validation, real mechanical faults were injected into a rotating motor (fan). 

![Grafana Dashboard and Telegram Alert](/images/Grafana_Pannel.jpeg)

**Key achievement:** After implementing a fault persistence logic (evaluation using 1-minute windows to ignore isolated noise), the system reached a **100% anomaly detection rate**.

> **Full Documentation:** For a detailed technical breakdown of the code, the business justification, and the comprehensive report of the physical tests, please check the `/docs` folder in this repository.

## How to run this project

1. **Hardware Setup:** Connect the MPU-6050 sensor to the Raspberry Pi using the I2C bus pins (SDA to pin 3, SCL to pin 5, VCC to 3.3V, and GND to GND). Rigidly attach the accelerometer to the casing of the motor you want to monitor to ensure pure vibration transmission.
2. **Environment Setup (Software):** Install InfluxDB (for storing time series) and Grafana (for the dashboard visualization) on your Raspberry Pi. Chat with the *BotFather* on Telegram to create a Bot and get your access Token.
3. **Baseline Capture (Dataset):** Turn on the machine and make sure it is running in an optimal (healthy) state. Run the `1_generate_dataset.py` script so the system starts recording the vibrations and physical features, creating your baseline dataset.
4. **Model Training:** Once you have enough healthy data, run the `2_train_model.py` script. The unsupervised learning algorithm will create its "bubble of normality" and save the trained model to local memory.
5. **Model Evaluation (Testing):** Before deploying the system, run `3_evaluate_model.py`. This script will test the generated model by checking its accuracy and confusion matrix, and verifying a low false alarm rate to ensure commercial viability.
6. **Production Deployment (Monitoring):** Run the main `4_monitor_production.py` script. The system will enter active listening mode, evaluating sliding time windows. If the vibration persistently strays outside the geometric bubble, the Telegram bot will trigger an alert to your mobile device.

## Proposed Future Improvements
* Transition to **Motor Current Signature Analysis (MCSA)** for a 100% non-invasive reading by measuring amperage at the electrical panel.
* Implementation of **Order Tracking** (dimensionless variables) to adapt the AI to variable-speed machinery.
