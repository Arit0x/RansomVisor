# 🔐 RansomVisor: Sistema Análisis y Predicción de Ransomware

![Banner RansomVisor](assets/portada_readme.png)

## 📋 Descripción General

RansomVisor es una plataforma avanzada de análisis y predicción de ataques ransomware desarrollada con Python y Streamlit. La aplicación integra análisis de datos en tiempo real, visualizaciones interactivas y modelos predictivos basados en machine learning para proporcionar inteligencia procesable sobre la evolución de amenazas ransomware.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1.4+-green.svg)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Características Principales

### 🔍 Análisis y Visualización
- **Dashboard Interactivo**: Estadísticas clave y métricas de ataques ransomware actualizadas
- **Tendencias Temporales**: Visualización avanzada de la evolución de ataques a lo largo del tiempo
- **Distribución Geográfica**: Mapas de calor y análisis regional de víctimas
- **Análisis Sectorial**: Impacto por sectores industriales con comparativas
- **Datos Actualizables**: Sistema para obtener los datos más recientes sobre ataques

### 🤖 Modelado Predictivo
- **Predicción Avanzada**: Modelo basado en Prophet con optimización automática
- **Variables Contextuales**: Incorporación de regresores externos (CVEs, eventos)
- **Detección de Outliers**: Identificación y manejo de valores atípicos
- **Validación Cruzada**: Evaluación robusta del rendimiento predictivo
- **Backtesting**: Prueba del modelo en datos históricos para validar precisión

### ⚙️ Características Técnicas
- **Arquitectura Modular**: Componentes independientes para fácil mantenimiento
- **Interfaz Intuitiva**: Diseño responsive optimizado para usuarios no técnicos
- **Caché Eficiente**: Optimización de rendimiento con Streamlit cache
- **Exportación de Resultados**: Descarga de predicciones y visualizaciones
- **Guía Integrada**: Documentación completa dentro de la aplicación

## 🏗️ Arquitectura del Sistema

RansomVisor está construido con una arquitectura modular que separa claramente la interfaz de usuario, la lógica de negocio y el procesamiento de datos:

```
RansomVisor/
├── 📊 Interfaz de Usuario (Streamlit)
│   ├── Páginas Múltiples
│   ├── Componentes Interactivos
│   └── Visualizaciones Dinámicas
│
├── 📈 Motor de Análisis
│   ├── Preprocesamiento de Datos
│   ├── Análisis Exploratorio
│   └── Cálculo de Métricas
│
└── 🧠 Motor Predictivo
    ├── Modelo Prophet
    ├── Optimización de Hiperparámetros
    ├── Calibración de Intervalos
    └── Evaluación de Rendimiento
```

## 🗂️ Estructura del Proyecto

```
web_final/
├── app.py                 # Punto de entrada principal
├── assets/                # Recursos estáticos (imágenes, CSS)
├── data/                  # Datos y snapshots
├── documentacion/         # Documentación detallada
├── modeling/              # Sistema de modelado predictivo
│   ├── data/              # Componentes de carga y preprocesamiento
│   ├── features/          # Generación de características y outliers
│   ├── models/            # Modelos predictivos (Prophet)
│   └── utils/             # Utilidades de visualización
├── pages/                 # Páginas de la aplicación
│   ├── home.py            # Dashboard principal
│   ├── tendencias.py      # Análisis de tendencias
│   ├── geografia.py       # Distribución geográfica
│   ├── alertas.py         # Sistema de alertas
│   ├── modelado_modular.py # Interfaz de predicción
│   └── sectores.py        # Análisis por sectores
├── utils.py               # Funciones de utilidad general
├── eda.py                 # Funciones de análisis exploratorio
├── sidebar.py             # Controles del sidebar
└── alerts.py              # Sistema de alertas y notificaciones
```

### 📊 Conjuntos de Datos Principales

El sistema utiliza los siguientes conjuntos de datos clave:

1. **Víctimas de Ransomware** (`modeling/victimas_ransomware_mod.json`):
   - Registro histórico de ataques ransomware
   - Información sobre víctimas, sectores y fechas
   - Fuente: Basado en datos de ransomware.live

2. **Vulnerabilidades CVE** (`modeling/cve_diarias_regresor_prophet.csv`):
   - Conteo diario de vulnerabilidades publicadas
   - Utilizado como regresor externo para mejorar predicciones
   - Proporciona contexto sobre el panorama de amenazas

3. **Snapshots Actualizables**:
   - `recentcyberattacks.json`: Ataques cibernéticos recientes
   - `recentvictims.json`: Víctimas recientes de ransomware
   - `flattened_ransomware_year.json`: Datos anuales procesados

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Pasos de Instalación

1. **Clonar el repositorio** (o descargar y descomprimir el ZIP):
   ```bash
   git clone https://github.com/tu-usuario/ransomvisor.git
   cd ransomvisor
   ```

2. **Crear un entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   
   # En Windows:
   venv\Scripts\activate
   
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencias Principales

```
streamlit>=1.24.0
pandas>=1.3.0
numpy>=1.20.0
prophet>=1.1.4
plotly>=5.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
folium>=0.12.0
holidays>=0.14
statsmodels>=0.13.0
openpyxl>=3.0.9
```

## 🎮 Uso

### Iniciar la Aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501` por defecto.

### Navegación

1. **Visión General** 📊: Dashboard principal con estadísticas clave y resumen de tendencias actuales.
2. **Tendencias** 📈: Análisis detallado de la evolución temporal de ataques ransomware.
3. **Geografía** 🌎: Distribución geográfica y análisis regional de víctimas.
4. **Alertas** 🟠: Sistema de notificación sobre nuevos ataques y grupos relevantes.
5. **Modelado** 🤖: Interfaz de predicción con modelo Prophet personalizable.
6. **Sectores** 🏭: Análisis del impacto por sectores industriales.

### Flujo de Trabajo de Modelado

1. **Carga de Datos**: Seleccione la fuente de datos de ransomware y variables exógenas.
2. **Preparación**: Configure parámetros de preprocesamiento y transformación de datos.
3. **Entrenamiento**: Ajuste los hiperparámetros del modelo Prophet según sus necesidades.
4. **Predicción**: Genere pronósticos para el período futuro deseado.
5. **Evaluación**: Valide el rendimiento del modelo mediante métricas de error y backtesting.

## 📊 Características del Modelo Predictivo

RansomVisor utiliza un modelo Prophet altamente personalizado para predecir ataques ransomware, con:

- **Detección Automática de Tendencias**: Identificación inteligente de cambios en patrones de ataques
- **Estacionalidad Múltiple**: Captura patrones diarios, semanales y anuales en datos de ataques
- **Regresores Externos**: Incorporación de variables como vulnerabilidades (CVEs) y eventos especiales
- **Optimización Bayesiana**: Ajuste automático de hiperparámetros para maximizar precisión
- **Calibración de Intervalos**: Intervalos de confianza precisos para evaluación de riesgos
- **Backtesting Configurable**: Prueba histórica para validar el rendimiento predictivo

## 📖 Documentación Adicional

La documentación completa del proyecto está disponible en la carpeta `documentacion/`:

- [Guía de Usuario](documentacion/guia_usuario.md): Instrucciones detalladas para usuarios finales
- [Documentación Técnica](documentacion/documentacion_tecnica.md): Arquitectura y detalles de implementación
- [Explicación del Modelo](documentacion/explicacion_modelo_ransomware.md): Fundamentos del modelo predictivo

Además, la aplicación incluye una **Guía de Usuario Interactiva** accesible desde la sección de Modelado.

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit, HTML/CSS, Plotly
- **Backend**: Python, Pandas, NumPy
- **Modelado**: Prophet, Scikit-learn, Holidays
- **Visualización**: Plotly, Matplotlib, Altair
- **Geoespacial**: Folium, GeoPandas

## 🙏 Agradecimientos

Este proyecto no habría sido posible sin los datos abiertos proporcionados por [Ransomware.live](https://www.ransomware.live). Agradecemos su labor y compromiso con la transparencia, que ha permitido analizar y visualizar el impacto global del ransomware de forma accesible y rigurosa.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un Fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/amazing-feature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`)
4. Sube los cambios a tu rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones, por favor contacta a:

- [Aridañy](mailto:arit0bx@gmail.com)
- [arit0x](https://github.com/Arit0x)

---

<p align="center">Desarrollado con ❤️ para la comunidad de ciberseguridad</p>