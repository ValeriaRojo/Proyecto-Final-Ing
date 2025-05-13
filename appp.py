import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image  # Para añadir logos/images
import plotly.graph_objects as go
from streamlit_option_menu import option_menu ### Agregado para hacer un menú de opciones

# Configuración inicial 
st.set_page_config(
    layout="wide",
    page_title="Airbnb Comparación Ciudades",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("css/style.css") #Uso de css (carga)

# Cache  con hash_funcs para objetos personalizados
@st.cache_data
def load_all_data():
    def clean_df(path):
        df = pd.read_csv(path)
        df['price'] = pd.to_numeric(df['price'].replace('[\$,]', '', regex=True), errors='coerce')
        return df.dropna(subset=['price'])

    df_mex = clean_df('data/Mexico_Datos_Limpios_2.csv')
    df_men = clean_df('data/spain_menorca_procesado_outliers (2) (1).csv')
    #df_osl = clean_df('data/oslo_procesado.csv')
    df_bos = clean_df('data/Boston_Limpio_Procesado.csv')

    return df_mex, df_men, df_bos

df_mex, df_men, df_bos = load_all_data()

# Sidebar con logo y secciones
with st.sidebar:
    # Puedes añadir un logo si tienes uno
    # logo = Image.open('logo.png')
    # st.image(logo, width=200)

    view = option_menu(
        menu_title = "Menú de Navegación",
        options = [
            "Inicio",
            "Vista General",
            "Análisis de Precios",
            "Regresión Lineal",
            "Regresión Logística"
        ],
        icons = [
            "house",
            "bar-chart",
            "graph-up",
            "activity"
        ],
        menu_icon = "cast",
        default_index = 0,
        styles = {
            "container": {"background-color": "#112D4E",  "font-family": "Segoe UI, sans-serif"},
            "icon": {"color": "#F9F7F7", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#86A788",
                "font-family": "Segoe UI, sans-serif"
            },
            "nav-link-selected": {
                "background-color": "#3F72AF",
                "color": "white",
                "font-family": "Segoe UI, sans-serif"
            },
        }
    )
    
    st.markdown("---")
    st.markdown("**Configuración general**")
    st.caption("Ajustes aplicables a todas las vistas")

# Función para mostrar métricas con estilo 
def display_metrics(metrics_dict, cols=3):
    columns = st.columns(cols)
    for i, (name, value) in enumerate(metrics_dict.items()):
        columns[i % cols].metric(label=name, value=value)

if view == "Inicio":

    from PIL import Image

    st.markdown("<h1 class='titulo-inicio'>Dashboard Comparativo de Airbnb</h1>", unsafe_allow_html=True)

    image = Image.open("img/portada.jpg")
    st.image(image, use_container_width= True)  # O el tamaño que gustes

    st.markdown("""
    <p style='font-size:18px; text-align:justify;'>
    Bienvenido al panel interactivo de análisis comparativo de alojamientos en <strong>Airbnb</strong>. Este dashboard te permite explorar, comparar y entender el comportamiento de variables clave como <em>precio, disponibilidad, calidad del anfitrión</em> y más, entre cuatro ciudades representativas: <strong>Boston, Menorca, Oslo y Ciudad de México</strong>.
    </p>

    <p style='font-size:18px; text-align:justify;'>
    A través de diferentes enfoques analíticos —como el análisis univariado, regresión lineal y regresión logística—, podrás identificar patrones y diferencias relevantes en la oferta de hospedaje de cada ciudad. Este análisis está diseñado para usuarios interesados en tomar decisiones informadas, ya sean turistas, anfitriones o investigadores del mercado turístico.
    </p>

    <p style='font-size:18px; text-align:justify;'>
    Explora cada sección en el menú lateral para comparar cómo se comportan los distintos factores en cada ciudad.
    </p>

    <p>¡Haz CLIC y empieza tu recorrido!</p>
    """, unsafe_allow_html=True)

if view == 'Vista General':
    st.title('🏠 Airbnb - Dashboard Analítico Comparativo')
    st.markdown("""
        <style>
        .big-font {
            font-size:16px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Comparación entre Ciudad de México y otra ciudad Airbnb</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Filtros Avanzados")

        ciudad_comparar = st.selectbox("Selecciona ciudad a comparar con México:", ["Menorca", "Boston"], index=0)

        df_comp = df_men if ciudad_comparar == "Menorca" else df_bos

        price_range = st.slider(
            'Rango de Precio (€)',
            float(df_comp['price'].min()),
            float(df_comp['price'].max()),
            (float(df_comp['price'].quantile(0.25)), float(df_comp['price'].quantile(0.75))),
            help="Selecciona el rango de precios a visualizar"
        )

        superhost_filter = st.selectbox(
            '¿Es Superhost?',
            options=['Todos', 'Sí', 'No'],
            index=0
        )

        room_type_filter = st.multiselect(
            'Tipo de Alojamiento',
            options=df_comp['room_type'].dropna().unique(),
            default=df_comp['room_type'].dropna().unique(),
            help="Selecciona uno o varios tipos de alojamiento"
        )

        show_data = st.checkbox("Mostrar datos filtrados", value=False)

    # Filtro
    filtered_df = df_comp.copy()
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]

    if superhost_filter != 'Todos':
        superhost_value = 't' if superhost_filter == 'Sí' else 'f'
        filtered_df = filtered_df[filtered_df['host_is_superhost'] == superhost_value]

    if room_type_filter:
        filtered_df = filtered_df[filtered_df['room_type'].isin(room_type_filter)]

    # Métricas clave
    metrics = {
        "Total Propiedades": len(filtered_df),
        "Precio Promedio": f"€{filtered_df['price'].mean():.2f}",
        "Rating Promedio": f"{filtered_df['review_scores_rating'].mean():.2f}",
        "Ocupación (30 días)": f"{filtered_df['availability_30'].mean():.1f} días"
    }
    display_metrics(metrics)

    tab1, tab2 = st.tabs(["📊 Distribución de Precios", "📈 Relaciones Clave"])

    ### TABLA 1
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("México")
            fig_mex = px.histogram(
                df_mex, 
                x='price', 
                nbins=50, 
                title='Distribución de Precios - México',
                color_discrete_sequence=['#1f77b4'],
                labels={'price': 'Precio (€)'}
            )
            fig_mex.update_layout(
                hovermode='x unified',
                xaxis_title="Precio (€)",
                yaxis_title="Número de propiedades"
            )
            st.plotly_chart(fig_mex, use_container_width=True)

        with col2:
            st.subheader(ciudad_comparar)
            fig_comp = px.histogram(
                filtered_df, 
                x='price', 
                nbins=50, 
                title=f'Distribución de Precios - {ciudad_comparar}',
                color_discrete_sequence=['#ff7f0e'],
                labels={'price': 'Precio (€)'}
            )
            fig_comp.update_layout(
                hovermode='x unified',
                xaxis_title="Precio (€)",
                yaxis_title="Número de propiedades"
            )
            st.plotly_chart(fig_comp, use_container_width=True)


        ###TABLA 2
        with tab2:
            col1, col2 = st.columns(2)

        with col1:
            st.subheader("México")
            fig_mex = px.scatter(
                df_mex,
                x='accommodates',
                y='price',
                color='room_type',
                size='bathrooms',
                hover_name='room_type',
                title='Capacidad vs Precio - México',
                labels={
                    'accommodates': 'Capacidad (personas)',
                    'price': 'Precio (€)',
                    'room_type': 'Tipo de habitación'
                }
            )
            st.plotly_chart(fig_mex, use_container_width=True)

        with col2: #Ya muestra 2 gráficas dependiendo lo que elija el usuario
            st.subheader(ciudad_comparar)
            fig_ciudad = px.scatter(
                filtered_df,
                x='accommodates',
                y='price',
                color='room_type',
                size='bathrooms',
                hover_name='room_type',
                title=f'Capacidad vs Precio - {ciudad_comparar}',
                labels={
                    'accommodates': 'Capacidad (personas)',
                    'price': 'Precio (€)',
                    'room_type': 'Tipo de habitación'
                }
            )
            st.plotly_chart(fig_ciudad, use_container_width=True)

        
        
elif view == 'Análisis de Precios':
    st.title('💰 Análisis de Precios')
    st.markdown("Explora cómo diferentes factores afectan el precio de los alojamientos.")
    
    # Widgets con descripciones
    with st.sidebar:
        st.header('Opciones de Análisis')
        analysis_type = st.radio(
            "Tipo de análisis",
            options=["Distribución", "Correlación", "Comparación"],
            index=0
        )
    
    if analysis_type == "Distribución":
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                'Variable para análisis',
                options=['room_type', 'host_is_superhost', 'host_response_time', 'accommodates', 'bathrooms'],
                index=0,
                help="Selecciona la variable para el eje X"
            )
            
        with col2:
            color_by = st.selectbox(
                'Variable de color',
                options=['room_type', 'host_is_superhost', None],
                index=0,
                help="Selecciona cómo colorear los datos"
            )
        
        fig = px.box(
            df, 
            x=x_axis, 
            y='price', 
            color=color_by,
            title=f'Distribución de Precios por {x_axis}',
            labels={'price': 'Precio (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Correlación":
        st.subheader('Correlación entre Variables')
        
        # Selección de variables numéricas
        numeric_features = ['price', 'accommodates', 'bathrooms', 'beds', 
                          'review_scores_rating', 'number_of_reviews']
        
        selected_features = st.multiselect(
            'Selecciona variables para el análisis de correlación',
            options=numeric_features,
            default=['price', 'accommodates', 'bathrooms']
        )
        
        if len(selected_features) >= 2:
            corr_matrix = df[selected_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matriz de Correlación",
                labels=dict(color="Correlación"),
                x=selected_features,
                y=selected_features,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selecciona al menos 2 variables para el análisis de correlación")
            
    elif analysis_type == "Comparación":
        st.subheader('Comparación de Precios')
        
        feature = st.selectbox(
            'Selecciona característica para comparar',
            options=['room_type', 'host_is_superhost', 'beds', 'accommodates'],
            index=0
        )
        
        fig = px.violin(
            df,
            x=feature,
            y='price',
            box=True,
            points="all",
            hover_data=df.columns,
            title=f'Distribución de Precios por {feature}'
        )
        st.plotly_chart(fig, use_container_width=True)
elif view == 'Regresión Lineal':
    st.title('📈 Modelo de Regresión Lineal')
    st.markdown("""
    Predice el precio basado en diferentes características del alojamiento.
    """)
    
    model_type = st.radio(
        'Tipo de modelo:',
        options=['Regresión Simple', 'Regresión Múltiple'],
        horizontal=True
    )
    
    if model_type == 'Regresión Simple':
        st.sidebar.header('Configuración - Regresión Simple')
        
        feature_options = [
            'accommodates', 'bathrooms', 'beds',
            'minimum_nights', 'number_of_reviews', 'availability_30'
        ]
        
        selected_feature = st.sidebar.selectbox(
            'Variable predictora:',
            options=feature_options,
            index=0
        )
        
        if selected_feature:
            temp_df = df[[selected_feature, 'price']].dropna()
            
            if len(temp_df) > 10:
                X = temp_df[[selected_feature]]
                y = temp_df['price']
                
                if X[selected_feature].dtype == 'object':
                    X = pd.get_dummies(X, drop_first=True)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                if X[selected_feature].dtype in ['int64', 'float64']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                
                fig = go.Figure()
                
                # Valores reales
                fig.add_trace(go.Scatter(
                    x=X_test.squeeze(),
                    y=y_test,
                    mode='markers',
                    name='Valores Reales',
                    marker=dict(color='#1f77b4', size=8),
                    opacity=0.7
                ))
                
                # Predicciones
                fig.add_trace(go.Scatter(
                    x=X_test.squeeze(),
                    y=y_pred,
                    mode='markers',
                    name='Predicciones',
                    marker=dict(color='#ff7f0e', size=8, symbol='x'),
                    opacity=0.7
                ))
                
                # Línea de tendencia
                if X[selected_feature].dtype in ['int64', 'float64']:
                    x_range = np.linspace(X_test.min(), X_test.max(), 100)
                    y_range = model.predict(x_range.reshape(-1, 1))
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        name='Línea de Regresión',
                        line=dict(color='#2ca02c', width=3)
                    ))
                
                fig.update_layout(
                    title=f'Relación entre {selected_feature} y Precio',
                    xaxis_title=selected_feature,
                    yaxis_title='Precio (€)',
                    hovermode='x unified'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Métricas del Modelo")
                    metrics = {
                        "R²": f"{r2_score(y_test, y_pred):.3f}",
                        #"MSE": f"{mean_squared_error(y_test, y_pred):.1f}",
                        #"RMSE": f"{np.sqrt(mean_squared_error(y_test, y_pred)):.1f}"
                    }
                    for name, value in metrics.items():
                        st.metric(label=name, value=value)
                    
                    if hasattr(model, 'coef_'):
                        st.write(f"Coeficiente: **{model.coef_[0]:.2f}**")
                    #st.write(f"Intercepto: **{model.intercept_:.2f}**")
                
                with col2:
                    st.subheader(f"Relación con {selected_feature}")
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("No hay suficientes datos para este análisis.")
    
    elif model_type == 'Regresión Múltiple':
        st.sidebar.header('Configuración - Regresión Múltiple')
        
        feature_options = [
            'accommodates', 'bathrooms', 'beds', 'review_scores_rating',
            'minimum_nights', 'number_of_reviews', 'availability_30',
            'room_type', 'host_is_superhost', 'instant_bookable',
            'bedrooms', 'host_total_listings_count'
        ]
        
        selected_features = st.sidebar.multiselect(
            'Variables predictoras:',
            options=feature_options,
            default=['accommodates', 'bathrooms', 'room_type']
        )
        
        if len(selected_features) >= 2:
            temp_df = df[selected_features + ['price']].dropna()
            
            X = temp_df[selected_features]
            y = temp_df['price']
            
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            if not X_encoded.empty:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                
                coef_df = pd.DataFrame({
                    'Variable': X_encoded.columns,
                    'Coeficiente': model.coef_,
                    #'Importancia': np.abs(model.coef_)
                }).sort_values('Coeficiente', ascending=False)
                
                top_feature = coef_df.iloc[0]['Variable']
                
                # Reconstruir X_test original para la variable principal
                if any(top_feature.startswith(col) for col in selected_features if df[col].dtype == 'object'):
                    # Para variables categóricas
                    cat_var = next(col for col in selected_features if top_feature.startswith(col))
                    x_test_values = X_test[:, X_encoded.columns.get_loc(top_feature)]
                    x_test_values = x_test_values * scaler.scale_[X_encoded.columns.get_loc(top_feature)] + scaler.mean_[X_encoded.columns.get_loc(top_feature)]
                else:
                    # Para variables numéricas
                    x_test_values = X_test[:, X_encoded.columns.get_loc(top_feature)]
                    x_test_values = x_test_values * scaler.scale_[X_encoded.columns.get_loc(top_feature)] + scaler.mean_[X_encoded.columns.get_loc(top_feature)]
                
                
                fig = go.Figure()
                
                # Valores reales
                fig.add_trace(go.Scatter(
                    x=x_test_values,
                    y=y_test,
                    mode='markers',
                    name='Valores Reales',
                    marker=dict(color='#1f77b4', size=8),
                    opacity=0.7
                ))
                
                # Predicciones
                fig.add_trace(go.Scatter(
                    x=x_test_values,
                    y=y_pred,
                    mode='markers',
                    name='Predicciones',
                    marker=dict(color='#ff7f0e', size=8, symbol='x'),
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title=f'Relación entre {top_feature} y Precio',
                    xaxis_title=top_feature,
                    yaxis_title='Precio (€)',
                    hovermode='x unified'
                )
                
                st.subheader('Resultados del Modelo')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
                    #st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.1f}")
                    #st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.1f}")
                    
                    st.subheader("Variables más importantes")
                    st.dataframe(coef_df.style.format({'Coeficiente': '{:.4f}', 'Importancia': '{:.4f}'}))
                
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Error en la preparación de datos.")
        else:
            st.warning("Selecciona al menos 2 variables para la regresión múltiple.")
elif view == 'Regresión Logística':
    st.title('📊 Modelo de Regresión Logística')
    st.markdown("""
    Predice si un alojamiento tiene un precio alto (por encima de la mediana) 
    basado en sus características.
    """)
    
    # Crear variable objetivo 
    price_median = df['price'].median()
    df['high_price'] = (df['price'] > price_median).astype(int)
    
    # Sidebar 
    with st.sidebar:
        st.header('Configuración del Modelo')
        
        feature_options = [
            'accommodates', 'bathrooms', 'beds', 'review_scores_rating',
            'minimum_nights', 'number_of_reviews', 'availability_30',
            'room_type', 'host_is_superhost'
        ]
        
        selected_features = st.multiselect(
            'Variables predictoras:',
            options=feature_options,
            default=['accommodates', 'bathrooms', 'host_is_superhost']
        )
        
        test_size = st.slider(
            'Tamaño del conjunto de prueba (%)',
            10, 40, 30
        )
    
    if selected_features:
        # Preparación de datos
        temp_df = df[selected_features + ['high_price']].dropna()
        X = temp_df[selected_features]
        y = temp_df['high_price']
        
        # Codificación
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        if not X_encoded.empty:
            # Escalado
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            
            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=test_size/100, 
                random_state=42,
                stratify=y
            )
            
            # Entrenamiento del modelo
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Resultados
            st.subheader('Evaluación del Modelo')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Precisión (Accuracy)", f"{accuracy_score(y_test, y_pred):.2f}")
                st.metric("Precisión (Precision)", f"{precision_score(y_test, y_pred):.2f}")
                st.metric("Sensibilidad (Recall)", f"{recall_score(y_test, y_pred):.2f}")
                st.metric("F1-Score", f"{2 * (precision_score(y_test, y_pred) * recall_score(y_test, y_pred)) / (precision_score(y_test, y_pred) + recall_score(y_test, y_pred)):.2f}")
            
            with col2:
                # Matriz de confusión mejorada
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicho", y="Real", color="Casos"),
                    x=['Bajo', 'Alto'],
                    y=['Bajo', 'Alto'],
                    title="Matriz de Confusión"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
      
            
            # Coeficientes del modelo
            st.subheader('Importancia de las Variables')
            
            if hasattr(model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Variable': X_encoded.columns,
                    'Coeficiente': model.coef_[0],
                    'Odds Ratio': np.exp(model.coef_[0])
                }).sort_values('Odds Ratio', ascending=False)
                
                fig_coef = px.bar(
                    coef_df,
                    x='Variable',
                    y='Odds Ratio',
                    color='Odds Ratio',
                    title='Odds Ratio de las Variables'
                )
                st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.error("Error en la preparación de datos.")
    else:
        st.warning("Selecciona al menos una variable predictora.")

