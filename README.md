# 🌎 IA para la detección y monitoreo de minería ilegal en Madre de Dios

Proyecto de **inteligencia artificial, teledetección y análisis geoespacial** para detectar y monitorear la expansión de la minería ilegal usando imágenes Sentinel-2, modelos Deep Learning y datos territoriales.

El caso piloto se desarrolla en **Madre de Dios**, provincia de **Tambopata**, distrito principal de **Inambari**, con un área aproximada de **944.64 km²**. La metodología tiene potencial de **escalabilidad a nivel nacional**.

---

## 🚀 Objetivo

Detectar, cuantificar y monitorear la minería ilegal entre **2019 y 2025**, generando información territorial útil para fiscalización ambiental, alertas tempranas y toma de decisiones públicas.

---

## 📍 Área de estudio

| Elemento | Descripción |
|---|---|
| Departamento | Madre de Dios |
| Provincia | Tambopata |
| Distrito principal | Inambari |
| Área aproximada | 944.64 km² |
| CRS | UTM Zone 19S / EPSG:32719 |
| Periodo | 2019–2025 |

---

## 🛰️ Datos utilizados

### Imágenes satelitales

- Sentinel-2
- Periodo: 2019–2025
- Meses priorizados: agosto a octubre

### Bandas

- B2
- B3
- B4
- B8
- B11
- B12

### Índices espectrales

- NDVI
- NDWI
- MNDWI
- NBR
- NDBI
- BSI

### Capas territoriales

- Límites administrativos
- Áreas naturales protegidas
- Zonas de amortiguamiento
- Comunidades nativas tituladas
- Ríos navegables
- Datos interoperables con GEO Perú / SNIG

---

## 🧠 Modelos entrenados

| Modelo | Descripción |
|---|---|
| U-Net baseline | Encoder ResNet34, 12 canales de entrada |
| SegFormer-B0 | Modelo principal, adaptado a 12 canales Sentinel-2 + índices |

---

## 📊 Resultados del modelo

| Modelo | IoU | Dice/F1 | Precisión | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| U-Net baseline | 0.8574 | 0.9090 | 0.9240 | 0.9189 | 0.9742 |
| SegFormer-B0 | 0.8603 | 0.9115 | 0.9119 | 0.9266 | 0.9756 |

**SegFormer-B0** fue seleccionado como modelo principal por su mejor desempeño global.

---

## 📈 Resultados temporales

| Año | Área minera detectada |
|---|---:|
| 2019 | 14,842.70 ha |
| 2020 | 15,422.58 ha |
| 2021 | 16,480.93 ha |
| 2022 | 17,321.73 ha |
| 2023 | 18,624.12 ha |
| 2024 | 20,346.87 ha |
| 2025 | 22,678.36 ha |

### Indicadores principales

- Incremento total: **7,835.66 ha**
- Variación total: **52.79%**
- Expansión nueva 2019–2025: **8,721.05 ha**
- Minería persistente: **13,957.31 ha**
- Huella acumulada: **24,741.49 ha**

---

## 🗺️ Hallazgos territoriales

| Capa territorial | Resultado principal |
|---|---:|
| Zonas de amortiguamiento | 4,233.19 ha de expansión nueva |
| Zonas de amortiguamiento | 15,683.85 ha de minería acumulada |
| Comunidades nativas | 178.37 ha de expansión nueva en Kotsimba |
| Ríos navegables | Presión en Río Manuani e Inambari |
| Áreas naturales protegidas | 9.32 ha de expansión nueva |

---

## ⚙️ Flujo metodológico

```text
Sentinel-2
↓
Cálculo de bandas e índices espectrales
↓
Etiquetado minería / no minería
↓
Entrenamiento U-Net y SegFormer
↓
Inferencia anual 2019–2025
↓
Vectorización de predicciones
↓
Análisis temporal
↓
Cruce con capas territoriales
↓
Mapas, indicadores y alertas tempranas
```
## 🌐 Escalabilidad

Aunque el piloto se desarrolló en Madre de Dios, la metodología puede escalarse a otras regiones del Perú afectadas por minería ilegal, deforestación o degradación ambiental. 

Para escalar el sistema se requiere:
* **Incorporar nuevas imágenes** provenientes de satélites como Sentinel-2.
* **Generar muestras locales** de entrenamiento adaptadas a la geografía de cada zona.
* **Ajustar o reentrenar el modelo** para asegurar una alta precisión.
* **Integrar capas territoriales** regionales para un mejor contexto geográfico.
* **Automatizar reportes** y la emisión de alertas tempranas.

---

## 🎯 Impacto Esperado

Este proyecto busca contribuir a:
* 🔍 **Mejorar la fiscalización ambiental** a través de tecnología precisa.
* 🚨 **Detectar nuevos frentes** de minería ilegal de manera temprana.
* 🌿 **Reducir riesgos socioambientales** asociados a la degradación del territorio.
* 📍 **Priorizar zonas críticas** para la intervención y control.
* 📊 **Fortalecer la toma de decisiones** en la gestión pública basadas en evidencia.
* 🏛️ **Impulsar la transformación digital** de las instituciones del Estado.

---

## 👤 Autor

* **Wilder T. Sebastian R.**

> 💡 *Proyecto desarrollado como propuesta para **GEOTÓN Perú 2026**, orientado al uso de datos georreferenciados, inteligencia artificial e innovación pública.*

---

## 📌 Mensaje Central

> 💬 **"La transformación digital no consiste solo en usar tecnología. Consiste en convertir datos en decisiones, evidencia en acción y territorio en prioridad pública."**
