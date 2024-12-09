import streamlit as st
import mysql.connector
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go

# Credenciales de la base de datos en la nube
DB_USER = "firetens_coronel"
DB_PASSWORD = "nCT,Wsvm5GMSC114Z%&O"
DB_HOST = "50.116.27.100"
DB_PORT = 3306
DB_NAME = "firetens_coronel"

# Conectar a la base de datos MySQL y obtener los datos
def get_data(product=None):
    try:
        # Conexión a la base de datos MySQL
        conexion = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conexion.cursor(dictionary=True)

        # Consulta SQL con filtros dinámicos
        query = """
        SELECT vp.fecha, vp.total, vpd.producto_id, p.nombre AS producto
        FROM ventasproductos vp
        INNER JOIN ventasproductodetalles vpd ON vp.id = vpd.ventasproducto_id
        INNER JOIN productos p ON vpd.producto_id = p.id
        WHERE 1=1
        """
        if product:
            query += f" AND p.id = {product}"

        cursor.execute(query)
        records = cursor.fetchall()
        df = pd.DataFrame(records)

        cursor.close()
        conexion.close()
        return df
    except mysql.connector.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return pd.DataFrame()

# Obtener nombres de productos desde la base de datos
def get_product_names():
    try:
        conexion = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conexion.cursor(dictionary=True)

        query = "SELECT id, nombre FROM productos"
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        conexion.close()

        # Retornar los productos como lista de tuplas (id, nombre)
        return [(record['id'], record['nombre']) for record in records]
    except mysql.connector.Error as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return []

# Procesar los datos para agrupar por día, semana o mes
def preprocess_data(df, interval='D'):
    try:
        df['fecha'] = pd.to_datetime(df['fecha'])  # Convertir fechas a formato datetime
        df.set_index('fecha', inplace=True)       # Usar fechas como índice

        # Convertir `total` a numérico y manejar errores
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df.dropna(subset=['total'], inplace=True)  # Eliminar valores nulos

        # Agrupar los datos según el intervalo seleccionado
        df_resampled = df['total'].resample(interval).sum()
        return df_resampled
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        return pd.Series()

# Entrenar el modelo ARIMA y realizar predicciones
def predict_sales(df, periods, interval_code):
    try:
        model = ARIMA(df, order=(1, 1, 1))  # Modelo ARIMA (p, d, q)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)  # Generar predicciones

        # Ajustar las fechas para la predicción
        if interval_code == 'M':  # Si el intervalo es mensual
            forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
        else:
            forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(1, unit=interval_code), periods=periods, freq=interval_code)
        
        forecast_df = pd.DataFrame({'Fecha': forecast_index, 'Predicción': forecast})
        return forecast_df
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return pd.DataFrame()

# Configuración de Streamlit
st.set_page_config(page_title="Predicción de Ventas", layout="centered")
st.title("📊 Predicción de Ventas por Producto")
st.write("Seleccione un producto para analizar las ventas y generar predicciones.")

# Obtener la lista de productos
products = get_product_names()
if products:
    product_options = {name: id for id, name in products}
    selected_product_name = st.selectbox("Selecciona un producto:", list(product_options.keys()))
    selected_product_id = product_options[selected_product_name]

    # Seleccionar el intervalo de predicción
    interval = st.radio("Selecciona el intervalo de predicción:", ("Día", "Semana", "Mes"))
    if interval == "Día":
        interval_code = 'D'
        periods = 7
    elif interval == "Semana":
        interval_code = 'W'
        periods = 4
    else:
        interval_code = 'M'
        periods = 4

    # Cargar y procesar los datos
    df = get_data(product=selected_product_id)
    if not df.empty:
        df_resampled = preprocess_data(df, interval=interval_code)

        if not df_resampled.empty:
            st.subheader(f"📈 Datos históricos de ventas ({selected_product_name})")
            fig = px.line(df_resampled, x=df_resampled.index, y=df_resampled, 
                          labels={'x': 'Fecha', 'y': 'Ventas'},
                          title=f"Ventas agrupadas por {interval.lower()}")
            fig.update_traces(line=dict(color='blue', width=3))
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar tabla de datos históricos
            st.subheader("📊 Tabla de Datos Históricos")
            st.dataframe(df_resampled)

            # Predicción de ventas
            forecast_df = predict_sales(df_resampled, periods, interval_code)
            if not forecast_df.empty:
                st.subheader(f"🔮 Predicción de ventas ({selected_product_name})")
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df_resampled.index, y=df_resampled, 
                                                  mode='lines', name='Histórico', line=dict(color='blue')))
                fig_forecast.add_trace(go.Scatter(x=forecast_df['Fecha'], y=forecast_df['Predicción'], 
                                                  mode='lines', name='Predicción', line=dict(color='orange')))
                fig_forecast.update_layout(title="Predicción de Ventas",
                                           xaxis_title="Fecha", yaxis_title="Ventas",
                                           template="plotly_white")
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Mostrar tabla de predicciones (solo Fecha y Predicción)
                st.subheader("📊 Tabla de Predicciones")
                forecast_df = forecast_df[['Fecha', 'Predicción']]  # Solo columnas necesarias
                st.dataframe(forecast_df)
        else:
            st.warning("⚠️ No se pudo procesar los datos. Verifique que la tabla contiene información válida.")
    else:
        st.warning("⚠️ No se encontraron datos para el producto seleccionado.")
else:
    st.warning("⚠️ No se pudo obtener la lista de productos desde la base de datos.")
