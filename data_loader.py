"""
=============================================================================
DATA LOADER - OLIST E-COMMERCE PROJECT
=============================================================================
Carga y caché de datos para toda la aplicación.
"""

import pandas as pd
from functools import lru_cache


# =============================================================================
# CARGA DE DATOS CON CACHÉ
# =============================================================================

@lru_cache(maxsize=1)
def load_data(file_path='df_oficial.xlsx'):
    """
    Carga el dataset principal con caché para evitar lecturas múltiples.

    Parameters:
    -----------
    file_path : str
        Ruta al archivo Excel

    Returns:
    --------
    pd.DataFrame
        Dataset completo de Olist
    """
    try:
        print("📂 Cargando datos...")
        df = pd.read_excel(file_path)
        print(f"✅ Datos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        print(f"📅 Período: 2016-2018")
        print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        return df
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None


def get_data_summary(df):
    """
    Retorna un resumen estadístico del dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset de Olist

    Returns:
    --------
    dict
        Diccionario con métricas clave
    """
    if df is None:
        return {}

    return {
        'total_orders': df['order_id'].nunique(),
        'total_customers': df['customer_unique_id'].nunique(),
        'total_sellers': df['seller_id'].nunique(),
        'total_products': df['product_id'].nunique(),
        'date_range': f"{df['order_purchase_timestamp'].min().strftime('%Y-%m-%d')} a {df['order_purchase_timestamp'].max().strftime('%Y-%m-%d')}",
        'avg_review_score': df['review_score'].mean(),
        'total_revenue': df['order_total_value'].sum(),
        'avg_delivery_time': df['delivery_time_days'].mean(),
        'on_time_rate': (df['on_time_delivery'].sum() / len(df) * 100)
    }


def get_variable_classifications(df):
    """
    Clasifica las variables del dataset por tipo.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset de Olist

    Returns:
    --------
    dict
        Diccionario con listas de variables por categoría
    """
    if df is None:
        return {}

    # Variables numéricas continuas
    continuas = [
        'price', 'freight_value', 'order_total_value', 'payment_value',
        'delivery_time_days', 'delivery_delay_days', 'freight_price_ratio',
        'product_weight_kg', 'product_volume_cm3', 'product_description_lenght',
        'product_name_lenght', 'product_height_cm', 'product_length_cm', 'product_width_cm'
    ]

    # Variables numéricas discretas
    discretas = [
        'review_score', 'payment_installments', 'product_photos_qty',
        'on_time_delivery', 'purchase_year', 'purchase_month', 'purchase_day_of_week', 'purchase_hour'
    ]

    # Variables categóricas
    categoricas = [
        'satisfaction_level', 'payment_type', 'product_category_name_english',
        'customer_state', 'seller_state', 'customer_city', 'seller_city'
    ]

    # Variables de fecha
    fechas = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date'
    ]

    # Variables identificadoras
    identificadores = [
        'order_id', 'customer_id', 'customer_unique_id', 'product_id',
        'seller_id', 'review_id', 'order_item_id'
    ]

    return {
        'continuas': [v for v in continuas if v in df.columns],
        'discretas': [v for v in discretas if v in df.columns],
        'categoricas': [v for v in categoricas if v in df.columns],
        'fechas': [v for v in fechas if v in df.columns],
        'identificadores': [v for v in identificadores if v in df.columns]
    }


def get_correlation_variables(df):
    """
    Retorna lista de variables para análisis de correlación.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset de Olist

    Returns:
    --------
    list
        Lista de variables numéricas para correlación
    """
    return [
        'review_score',
        'delivery_time_days',
        'delivery_delay_days',
        'on_time_delivery',
        'price',
        'freight_value',
        'order_total_value',
        'payment_installments',
        'freight_price_ratio',
        'product_photos_qty',
        'product_weight_kg',
        'product_volume_cm3'
    ]


# =============================================================================
# FUNCIÓN DE INICIALIZACIÓN
# =============================================================================

def initialize_data():
    """
    Función principal para inicializar y validar datos.

    Returns:
    --------
    tuple
        (df, summary, classifications)
    """
    df = load_data()

    if df is not None:
        summary = get_data_summary(df)
        classifications = get_variable_classifications(df)
        return df, summary, classifications

    return None, {}, {}


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == '__main__':
    df, summary, classifications = initialize_data()

    if df is not None:
        print("\n📊 RESUMEN DE DATOS:")
        for key, value in summary.items():
            print(f"   • {key}: {value}")

        print("\n📋 CLASIFICACIÓN DE VARIABLES:")
        for cat, vars_list in classifications.items():
            print(f"   • {cat}: {len(vars_list)} variables")
