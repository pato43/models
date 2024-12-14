import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

# Configuración de página
st.set_page_config(page_title="Portafolio de Proyectos de Ciencia de Datos", layout="wide", initial_sidebar_state="expanded")
st.title("Explora Nuestros Proyectos de Ciencia de Datos en México")
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
    **Proyecto:**
    Colaboramos con el municipio de Tenancingo para analizar la deserción escolar en nivel secundaria. Utilizamos modelos predictivos que relacionaron factores socioeconómicos y rendimiento académico, ayudando a identificar a estudiantes en riesgo.

    **Impacto:**
    Este proyecto permitió diseñar intervenciones educativas dirigidas, logrando reducir la deserción escolar en un 15% en un año académico.
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

# Pestaña 2: Segmentación Bancaria
with tabs[1]:
    st.subheader("Segmentación de Clientes para Optimización de Servicios Bancarios")
    st.markdown("""
    **Proyecto:**
    Realizamos un análisis de segmentación para Coppel, clasificando a sus clientes en grupos homogéneos con base en su comportamiento financiero. 

    **Impacto:**
    Este análisis permitió a Coppel diseñar estrategias de marketing más efectivas, incrementando las ventas cruzadas en un 20% en los segmentos identificados.
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

# Pestaña 3: Resultados Electorales
with tabs[2]:
    st.subheader("Predicción de Resultados Electorales en las Elecciones de EE. UU.")
    st.markdown("""
    **Proyecto:**
    Implementamos un modelo de predicción basado en datos históricos y encuestas para analizar las pasadas elecciones presidenciales en EE. UU. 

    **Impacto:**
    Ayudamos a identificar tendencias clave que permitieron orientar estrategias de campaña con mayor precisión, logrando un incremento en la participación del electorado objetivo.
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

# Pestaña 4: Análisis de Accidentes
with tabs[3]:
    st.subheader("Análisis de Accidentes de Tránsito en la Zona Metropolitana de la CDMX")
    st.markdown("""
    **Proyecto:**
    Analizamos datos de accidentes de tránsito en la CDMX para identificar zonas críticas y patrones recurrentes.

    **Impacto:**
    Este estudio permitió a las autoridades implementar medidas preventivas en las zonas de mayor riesgo, reduciendo los accidentes severos en un 12% en seis meses.
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
    st.subheader("Análisis de Demanda de Vivienda en la CDMX")
    st.markdown("""
    **Proyecto:**
    Evaluamos la demanda habitacional en la CDMX, analizando cómo factores como ubicación y tamaño afectan los precios de vivienda.

    **Impacto:**
    Los resultados ayudaron a los desarrolladores a identificar áreas de alta demanda, optimizando la oferta de vivienda y aumentando las ventas en un 25% en las zonas priorizadas.
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
    st.subheader("Análisis de Ventas Minoristas para Optimización de Inventarios")
    st.markdown("""
    **Proyecto:**
    Colaboramos con una cadena minorista para analizar patrones de ventas y optimizar la gestión de inventarios. 

    **Impacto:**
    La implementación de este análisis redujo la sobreproducción en un 18%, mejorando la eficiencia operativa.
    """)
    data_ventas = pd.DataFrame({
        "Producto": [f"Producto {i}" for i in range(1, 11)],
        "Ventas": np.random.randint(50, 500, 10)
    })
    fig = px.bar(
        data_ventas, 
        x="Producto", 
        y="Ventas", 
        title="Ventas por Producto"
    )
    st.plotly_chart(fig)

# Pestaña 7: Riesgo de Crédito
with tabs[6]:
    st.subheader("Evaluación de Riesgo de Crédito para Instituciones Financieras")
    st.markdown("""
    **Proyecto:**
    Desarrollamos un modelo de clasificación para evaluar el riesgo de crédito de clientes, utilizando datos históricos financieros.

    **Impacto:**
    Las instituciones lograron reducir la morosidad en un 10% mediante estrategias de mitigación basadas en este modelo.
    """)
    data_credito = pd.DataFrame({
        "Score de Crédito": np.random.uniform(300, 850, 100),
        "Probabilidad de Incumplimiento": np.random.uniform(0, 1, 100)
    })
    fig = px.scatter(
        data_credito, 
        x="Score de Crédito", 
        y="Probabilidad de Incumplimiento", 
        color="Probabilidad de Incumplimiento", 
        color_continuous_scale="Reds",
        title="Riesgo de Crédito"
    )
    st.plotly_chart(fig)

# Pestaña 8: Consumo Energético
with tabs[7]:
    st.subheader("Predicción de Consumo Energético en Hogares")
    st.markdown("""
    **Proyecto:**
    Implementamos un modelo predictivo para analizar el consumo energético en hogares y proponer estrategias de ahorro energético.

    **Impacto:**
    Las estrategias derivadas permitieron reducir el consumo promedio en un 15%, beneficiando tanto a los usuarios como al medio ambiente.
    """)
    data_energia = pd.DataFrame({
        "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
        "Consumo (kWh)": np.random.randint(100, 500, 12)
    })
    fig = px.line(
        data_energia, 
        x="Mes", 
        y="Consumo (kWh)", 
        title="Consumo Energético Mensual"
    )
    st.plotly_chart(fig)

# Pestaña 9: Salud Pública
with tabs[8]:
    st.subheader("Análisis de Factores de Salud Pública")
    st.markdown("""
    **Proyecto:**
    Evaluamos indicadores de salud pública como obesidad y diabetes para identificar tendencias en comunidades vulnerables.

    **Impacto:**
    Este análisis apoyó la implementación de programas de salud enfocados, reduciendo la incidencia de enfermedades en un 8% en un año.
    """)
    data_salud = pd.DataFrame({
        "Comunidad": [f"Comunidad {i}" for i in range(1, 11)],
        "Tasa de Obesidad": np.random.uniform(10, 50, 10)
    })
    fig = px.bar(
        data_salud, 
        x="Comunidad", 
        y="Tasa de Obesidad", 
        title="Tasa de Obesidad por Comunidad"
    )
    st.plotly_chart(fig)

# Pestaña 10: Movilidad Urbana
with tabs[9]:
    st.subheader("Optimización de Movilidad Urbana en CDMX")
    st.markdown("""
    **Proyecto:**
    Analizamos datos de movilidad urbana para proponer mejoras en la infraestructura vial y optimización del transporte público.

    **Impacto:**
    Las medidas propuestas redujeron el tiempo promedio de desplazamiento en un 12% y mejoraron la experiencia del usuario.
    """)
    data_movilidad = pd.DataFrame({
        "Hora": np.arange(24),
        "Flujo de Vehículos": np.random.randint(100, 1000, 24)
    })
    fig = px.area(
        data_movilidad, 
        x="Hora", 
        y="Flujo de Vehículos", 
        title="Flujo Vehicular por Hora"
    )
    st.plotly_chart(fig)
