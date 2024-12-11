import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

# Configuración de página
st.set_page_config(page_title="Curso de Data Science - Ejemplos", layout="wide", initial_sidebar_state="expanded")
st.title("Explora los Modelos y Aplicaciones de Ciencia de Datos en México")
st.markdown("---")

# Paleta de colores y configuración para modo oscuro
st.markdown("<style>body {background-color: #1e1e1e; color: white;} .stButton button {color: #1e1e1e; background-color: #a3d8f4;} .stTabs [role='tablist'] button {background-color: #2c2c2c; color: white;}</style>", unsafe_allow_html=True)

# Organizar en pestañas
tabs = st.tabs([
    "Deserción Escolar", 
    "Segmentación Bancaria", 
    "Resultados Electorales", 
    "Análisis de Accidentes", 
    "Demanda de Vivienda", 
    "Ventas Minoristas", 
    "Riesgo de Crédito", 
    "Consumo Energético", 
    "Salud Pública", 
    "Movilidad Urbana"
])

# Pestaña 1: Predicción de Deserción Escolar
with tabs[0]:
    st.subheader("Predicción de Deserción Escolar en Nivel Secundaria")
    st.markdown("""
    **Descripción:**
    Este ejemplo analiza la deserción escolar en nivel secundaria mediante el uso de modelos predictivos que relacionan factores socioeconómicos y rendimiento académico con la probabilidad de abandono escolar.
    
    **Qué se verá:**
    Un análisis visual que representa cómo el rendimiento académico afecta la probabilidad de deserción, destacando grupos de estudiantes con alto riesgo.
    
    **Qué se aprenderá:**
    - Aplicación de modelos estadísticos para evaluar probabilidades.
    - Técnicas de filtrado de datos relevantes para identificar patrones de comportamiento.
    - Uso de visualizaciones para comunicar hallazgos clave en el contexto educativo.
    
    **Temas relevantes:**
    - Estadística descriptiva e inferencial.
    - Modelado probabilístico.
    - Aplicación de análisis de correlación en contextos sociales.
    """)
    np.random.seed(42)
    data = pd.DataFrame({
        "Rendimiento Académico": np.random.uniform(0, 10, 100),
        "Probabilidad de Deserción": np.random.uniform(0, 1, 100)
    })
    fig = px.scatter(
        data, 
        x="Rendimiento Académico", 
        y="Probabilidad de Deserción", 
        color="Probabilidad de Deserción", 
        color_continuous_scale="Viridis",
        title="Probabilidad de Deserción vs Rendimiento Académico"
    )
    st.plotly_chart(fig)
    
    prob_filter = st.slider("Filtrar estudiantes con probabilidad mayor a", 0.0, 1.0, 0.5, step=0.1)
    filtered_data = data[data["Probabilidad de Deserción"] > prob_filter]
    st.markdown(f"### Estudiantes con probabilidad > {prob_filter}")
    st.write(filtered_data)

# Pestaña 2: Segmentación Bancaria
with tabs[1]:
    st.subheader("Segmentación de Clientes para Optimización de Servicios Bancarios")
    st.markdown("""
    **Descripción:**
    El objetivo es clasificar a los clientes bancarios en grupos homogéneos con base en su comportamiento financiero, utilizando técnicas de agrupamiento no supervisado.
    
    **Qué se verá:**
    Representación gráfica de los segmentos bancarios según sus ingresos y gastos, identificando patrones de comportamiento financiero dentro de cada grupo.
    
    **Qué se aprenderá:**
    
    - Principios de agrupamiento y distancia euclidiana.
    - Aplicación de métodos no supervisados para segmentación de datos.
    - Análisis de perfiles de consumidores para toma de decisiones.
    
    **Temas relevantes:**
    - Algoritmos de clustering (K-Means, jerárquico).
    - Reducción de dimensionalidad para análisis exploratorio.
    - Aplicación de teoría de decisiones en marketing.
    """)
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    data_cluster = pd.DataFrame(X, columns=["Ingresos", "Gasto"])
    data_cluster["Segmento"] = y

    fig = px.scatter(
        data_cluster, 
        x="Ingresos", 
        y="Gasto", 
        color=data_cluster["Segmento"].astype(str), 
        title="Segmentación de Clientes Bancarios",
        labels={"color": "Segmento"}
    )
    st.plotly_chart(fig)

    segmento = st.selectbox("Selecciona un segmento para explorar detalles:", [f"Segmento {i}" for i in range(4)])
    st.markdown(f"### Información del {segmento}")
    st.write(data_cluster[data_cluster["Segmento"] == int(segmento.split()[-1])])

# Pestaña 3: Resultados Electorales
with tabs[2]:
    st.subheader("Predicción de Resultados Electorales en Municipios")
    st.subheader("Segmentación de Clientes para Optimización de Servicios Bancarios")
    st.markdown("""
    **Descripción:**
    Este análisis se centra en modelar y proyectar tendencias de preferencia electoral a lo largo de una campaña, basándose en datos históricos y encuestas.
    
    **Qué se verá:**
    Gráficos temporales que muestran la evolución de las preferencias electorales por partido, destacando picos y tendencias significativas durante la campaña.
    
    **Qué se aprenderá:**
    
    - Modelado de datos temporales para identificar tendencias.
    - Análisis de factores influyentes en comportamientos sociales.
    - Técnicas de proyección para estimar resultados futuros.
    
    **Temas relevantes:**
    - Series de tiempo en estadística.
    - Análisis multivariable aplicado a datos sociopolíticos.
    - Métodos de interpolación y extrapolación.
    """)
    x = np.arange(10)
    partido_A = np.random.randint(30, 70, size=10)
    partido_B = 100 - partido_A
    data_electoral = pd.DataFrame({"Día": x, "Partido A": partido_A, "Partido B": partido_B})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_electoral["Día"], y=data_electoral["Partido A"], mode='lines+markers', name="Partido A", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data_electoral["Día"], y=data_electoral["Partido B"], mode='lines+markers', name="Partido B", line=dict(color="red")))
    fig.update_layout(title="Evolución de la Preferencia Electoral", xaxis_title="Día de Campaña", yaxis_title="Porcentaje de Preferencia")
    st.plotly_chart(fig)

    day_filter = st.slider("Selecciona el día de campaña", 0, 9, 5)
    st.markdown(f"### Preferencias del día {day_filter}")
    st.write(data_electoral[data_electoral["Día"] == day_filter])

# Pestaña 4: Análisis de Accidentes
with tabs[3]:
    st.subheader("Análisis de Accidentes de Tránsito")
    st.markdown("""
    **Descripción:**
    Se identifican zonas críticas de accidentes de tránsito a través de un análisis geoespacial basado en datos de gravedad y ubicación.
    
    **Qué se verá:**
    Una representación espacial que relaciona la ubicación geográfica de los accidentes con su gravedad, destacando zonas prioritarias para intervención.
    
    **Qué se aprenderá:**
    
    - Análisis geoespacial para evaluar patrones de incidencia.
    - Uso de datos categóricos en visualizaciones.
    - Interpretación de distribuciones espaciales en problemáticas urbanas.
    
    **Temas relevantes:**
    - Estadística espacial aplicada.
    - Modelado categórico para análisis descriptivo.
    - Optimización de recursos para diseño urbano.
    """)
    data_accidentes = pd.DataFrame({
        "Latitud": np.random.uniform(19.0, 19.5, 100),
        "Longitud": np.random.uniform(-99.5, -99.0, 100),
        "Gravedad": np.random.choice(["Leve", "Moderado", "Severo"], 100)
    })
    fig = px.scatter_mapbox(
        data_accidentes, 
        lat="Latitud", 
        lon="Longitud", 
        color="Gravedad", 
        mapbox_style="carto-darkmatter",
        title="Mapa de Accidentes de Tránsito",
        zoom=10
    )
    st.plotly_chart(fig)

# Pestaña 5: Demanda de Vivienda
with tabs[4]:
    st.subheader("Análisis de Demanda de Vivienda")
    st.markdown("""
    **Descripción:**
    Este análisis evalúa las características de la demanda habitacional, explorando cómo factores como ubicación y tamaño afectan los precios.
    
    **Qué se verá:**
    Un análisis que muestra la relación entre características clave de las propiedades y su valoración económica, destacando tendencias regionales.
    
    **Qué se aprenderá:**
    
    - Análisis de regresión en contextos económicos.
    - Identificación de variables influyentes en los precios de vivienda.
    - Uso de datos históricos para inferencias económicas.
    
    **Temas relevantes:**
    - Regresión multivariable y econometría.
    - Estadística descriptiva aplicada a bienes raíces.
    - Visualización de datos multidimensionales.
    """)
    data_vivienda = pd.DataFrame({
        "Precio": np.random.uniform(500000, 3000000, 100),
        "Superficie": np.random.uniform(50, 300, 100),
        "Ubicación": np.random.choice(["Centro", "Periferia", "Suburbios"], 100)
    })
    fig = px.scatter(
        data_vivienda, 
        x="Superficie", 
        y="Precio", 
        color="Ubicación", 
        title="Demanda de Vivienda por Ubicación"
    )
    st.plotly_chart(fig)

# Pestaña 6: Ventas Minoristas
with tabs[5]:
    st.subheader("Análisis de Ventas Minoristas")
    st.markdown("""
    **Descripción:**
    Se proyectan tendencias de ventas a futuro mediante el análisis de datos históricos de ventas y series temporales.
    
    **Qué se verá:**
    Una representación de las fluctuaciones en las ventas de distintas categorías, destacando patrones estacionales y anomalías.
    
    **Qué se aprenderá:**
    
    - Análisis de series temporales para modelado predictivo.
    - Identificación de estacionalidad y tendencias en datos de consumo.
    - Técnicas de preprocesamiento para mejorar proyecciones.
    
    **Temas relevantes:**
    - Modelos ARIMA, PROPHET, XGBOOST y descomposición de series temporales.
    - Análisis de patrones estacionales y de tendencia.
    - Optimización basada en predicciones.
    """)
    fechas = pd.date_range(start="2024-01-01", periods=100)
    ventas = np.random.poisson(100, 100)
    data_ventas = pd.DataFrame({"Fecha": fechas, "Ventas": ventas})
    fig = px.line(data_ventas, x="Fecha", y="Ventas", title="Tendencia de Ventas Diarias")
    st.plotly_chart(fig)

# Pestaña 7: Riesgo de Crédito
with tabs[6]:
    st.subheader("Evaluación de Riesgo Crediticio")
    st.markdown("""
    **Descripción:**
    Se evalúa el riesgo crediticio de clientes mediante un análisis de sus características financieras y comportamientos históricos.
    
    **Qué se verá:**
    Una visualización que segmenta a los clientes según su nivel de riesgo, permitiendo identificar perfiles específicos.
    
    **Qué se aprenderá:**
    
    - Modelado de datos categóricos para análisis de riesgo.
    - Técnicas de evaluación y segmentación de riesgos financieros.
    - Interpretación de datos para decisiones en préstamos.
    
    **Temas relevantes:**
    - Árboles de decisión y modelos probabilísticos.
    - Análisis de distribuciones para datos financieros.
    - Fundamentos de scoring crediticio.
    """)
    data_credito = pd.DataFrame({
        "Ingresos": np.random.uniform(10000, 100000, 100),
        "Deuda": np.random.uniform(1000, 50000, 100),
        "Riesgo": np.random.choice(["Bajo", "Moderado", "Alto"], 100)
    })
    fig = px.scatter(
        data_credito, 
        x="Ingresos", 
        y="Deuda", 
        color="Riesgo", 
        title="Evaluación del Riesgo Crediticio"
    )
    st.plotly_chart(fig)

# Pestaña 8: Consumo Energético
with tabs[7]:
    st.subheader("Predicción de Consumo Energético")
    st.markdown("""
    **Descripción:**
    Se analizan patrones de consumo energético en función de horarios y temporadas, identificando oportunidades de optimización.
    
    **Qué se verá:**
    Gráficos que representan las variaciones en el consumo energético a lo largo del tiempo y su distribución entre sectores.
    
    **Qué se aprenderá:**
    
    - Análisis de patrones de uso en datos horarios.
    - Técnicas para detectar ineficiencias energéticas.
    - Representación gráfica de distribuciones de consumo.
    
    **Temas relevantes:**
    - Series temporales aplicadas a recursos.
    - Análisis de densidad y variabilidad.
    - Optimización en gestión de recursos energéticos.
    """)
    sectores = ["Residencial", "Comercial", "Industrial"]
    data_energia = pd.DataFrame({
        "Sector": np.random.choice(sectores, 100),
        "Consumo": np.random.uniform(100, 1000, 100)
    })
    fig = px.box(data_energia, x="Sector", y="Consumo", color="Sector", title="Consumo Energético por Sector")
    st.plotly_chart(fig)

# Pestaña 9: Salud Pública
with tabs[8]:
    st.subheader("Análisis de Salud Pública")
    st.markdown("""
    **Descripción:**
    Este análisis examina la prevalencia de enfermedades en distintas regiones, buscando identificar áreas críticas para intervenciones sanitarias.
    
    **Qué se verá:**
    Una visualización que muestra la concentración geográfica de casos y su relación con factores socioeconómicos.
    
    **Qué se aprenderá:**
    
    - Análisis estadístico en datos de salud pública.
    - Métodos para detectar correlaciones en distribuciones espaciales.
    - Evaluación de impactos sociales en problemáticas sanitarias.
    
    **Temas relevantes:**
    - Estadística multivariable aplicada a datos de salud.
    - Modelos epidemiológicos descriptivos.
    - Gestión de datos categóricos y numéricos.
    """)
    enfermedades = ["Respiratorias", "Digestivas", "Cardíacas"]
    data_salud = pd.DataFrame({
        "Enfermedad": np.random.choice(enfermedades, 100),
        "Casos": np.random.poisson(50, 100)
    })
    fig = px.bar(data_salud, x="Enfermedad", y="Casos", color="Enfermedad", title="Casos por Tipo de Enfermedad")
    st.plotly_chart(fig)

# Pestaña 10: Movilidad Urbana
with tabs[9]:
    st.subheader("Análisis de Movilidad Urbana")
    st.markdown("""
    **Descripción:**
    Se analizan patrones de tráfico y movilidad urbana para identificar cuellos de botella y mejorar la planeación de rutas.
    
    **Qué se verá:**
    Gráficos que representan trayectorias, densidad de tránsito y flujos en áreas congestionadas.
    
    **Qué se aprenderá:**
    
    - Análisis de datos de flujo en redes de transporte.
    - Identificación de patrones de uso de infraestructura.
    - Aplicación de datos para mejorar la movilidad urbana.
    
    **Temas relevantes:**
    - Teoría de redes y flujos en transporte.
    - Análisis de densidad y nodos críticos.
    - Optimización en planeación urbana.
    """)
    data_movilidad = pd.DataFrame({
        "Hora": np.tile(range(24), 10),
        "Pasajeros": np.random.poisson(100, 240)
    })
    fig = px.line(
        data_movilidad.groupby("Hora").sum().reset_index(), 
        x="Hora", 
        y="Pasajeros", 
        title="Flujo de Pasajeros por Hora"
    )
    st.plotly_chart(fig)
