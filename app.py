import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go  # <--- NEW: Added for the Candlestick Chart

# --- PAGE SETUP ---
st.set_page_config(page_title="MatRisk AI Dashboard", layout="wide")
st.title("🚀 Excavate: Material & Market Intelligence")

# --- LOAD ALL DATA & MODELS ---
# --- LOAD ALL DATA & MODELS (OPTIMIZED) ---
@st.cache_data
def load_datasets():
    # Cache_data is used for dataframes so they load instantly
    df_m = pd.read_csv('data/cleaned_materials.csv')
    df_f = pd.read_csv('data/merged_financials.csv')
    df_p = pd.read_csv('data/DS5.csv')
    return df_m, df_f, df_p

@st.cache_resource
def load_ml_model():
    # Cache_resource is used for AI models so they stay in memory
    with open('data/task1_rf_model.pkl', 'rb') as f1:
        return pickle.load(f1)

try:
    df_materials, df_finance, df_prices = load_datasets()
    model_task1 = load_ml_model()
except Exception as e:
    st.error(f"Error loading files. Details: {e}")
    st.stop()

# --- CREATE 3 TABS FOR UI ---
tab1, tab2, tab3 = st.tabs([
    "💎 Task 1: Material Simulator", 
    "📈 Task 2: Market Radar", 
    "🧪 Bonus: Alloy Cost Optimizer"
])

# === TAB 1: MATERIAL SIMULATOR (Task 1) ===
with tab1:
    st.header("Predict Material Quality Index (MQI)")
    col1, col2 = st.columns(2)
    
    with col1:
        energy = st.slider("Formation Energy (eV)", -5.0, -0.1, -1.5)
        bulk_mod = st.slider("Bulk Modulus (GPa)", 10.0, 400.0, 100.0)
        shear_mod = st.slider("Shear Modulus (GPa)", 10.0, 300.0, 50.0)
        
    with col2:
        poisson = st.slider("Poisson's Ratio", -0.99, 0.49, 0.25)
        density = st.slider("Density (g/cm³)", 0.5, 20.0, 5.0)
        
    if st.button("🧠 Predict MQI Score"):
        input_data = pd.DataFrame({
            'formation_energy_per_atom_eV': [energy],
            'bulk_modulus_GPa': [bulk_mod],
            'shear_modulus_GPa': [shear_mod],
            'poisson_ratio': [poisson],
            'density_g_cm3': [density]
        })
        prediction = model_task1.predict(input_data)[0]
        st.success(f"### Predicted MQI: {prediction:.2f}")

    # --- UPGRADE: 3D MATERIAL UNIVERSE ---
    st.divider()
    st.subheader("🌌 3D Material Universe")
    st.markdown("Rotate and zoom to explore how density and moduli affect the MQI.")
    
    fig_3d = px.scatter_3d(
        df_materials, 
        x='density_g_cm3', 
        y='bulk_modulus_GPa', 
        z='shear_modulus_GPa',
        color='MQI',
        hover_data=['formula'],
        color_continuous_scale='Viridis',
        opacity=0.7
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig_3d, use_container_width=True)

# === TAB 2: MARKET RADAR (Task 2) ===
with tab2:
    st.header("Commodity Volatility & Supply Signals")
    commodity_list = df_finance['commodity'].unique()
    selected_asset = st.selectbox("Select a Commodity to Analyze:", commodity_list)
    asset_data = df_finance[df_finance['commodity'] == selected_asset]
    
    # --- UPGRADE: CANDLESTICK CHART ---
    st.subheader(f"📊 Live Price Action for {selected_asset}")
    
    candlestick = go.Figure(data=[go.Candlestick(
        x=asset_data['date'],
        open=asset_data['open'],
        high=asset_data['high'],
        low=asset_data['low'],
        close=asset_data['close'],
        increasing_line_color='cyan', decreasing_line_color='gray'
    )])
    candlestick.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(t=10, b=10))
    st.plotly_chart(candlestick, use_container_width=True)
    
    st.divider()
    
    # Original Scatter Plot
    st.subheader(f"Market Impact of Supply Chain Risks")
    fig = px.scatter(
        asset_data, x='supply_disruption_prob', y='daily_return', color='mqi',
        labels={'supply_disruption_prob': 'Risk of Supply Disruption', 'daily_return': 'Daily Return (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

# === TAB 3: ALLOY COST OPTIMIZER (Bonus Task) ===
with tab3:
    st.header("Inverse Design: Economic Feasibility")
    st.markdown("Mix different elements to create a new alloy and calculate its theoretical cost.")
    
    element_col = df_prices.columns[0] 
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Element 1")
        elem1 = st.selectbox("Select Element A", df_prices[element_col].unique(), key='e1')
        pct1 = st.slider("Percentage (%)", 0, 100, 60, key='p1')
        
    with col_b:
        st.subheader("Element 2")
        elem2 = st.selectbox("Select Element B", df_prices[element_col].unique(), key='e2')
        pct2 = st.slider("Percentage (%)", 0, 100, 40, key='p2')

    total_pct = pct1 + pct2
    if total_pct != 100:
        st.warning(f"Note: Your percentages currently add up to {total_pct}%. They should ideally equal 100%.")

    if st.button("💰 Calculate Alloy Cost"):
        price1 = df_prices[df_prices[element_col] == elem1]['price_usd_per_kg'].values[0]
        price2 = df_prices[df_prices[element_col] == elem2]['price_usd_per_kg'].values[0]
        
        total_cost = (price1 * (pct1 / 100)) + (price2 * (pct2 / 100))
        
        st.success(f"### Estimated Cost of new {elem1}-{elem2} Alloy: ${total_cost:.2f} per kg")
        st.info("💡 **Innovation Insight:** If you can achieve a high MQI (from Tab 1) while keeping this cost lower than standard market prices, you have a highly profitable material!")                
        
        # --- UPGRADE: EXPORT TO CSV BUTTON ---
        alloy_recipe = pd.DataFrame({
            'Element': [elem1, elem2],
            'Percentage': [f"{pct1}%", f"{pct2}%"],
            'Price per kg': [f"${price1:.2f}", f"${price2:.2f}"],
            'Total Alloy Cost': [f"${total_cost:.2f}", ""]
        })
        
        csv_data = alloy_recipe.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Alloy Recipe (CSV)",
            data=csv_data,
            file_name=f"custom_{elem1}_{elem2}_alloy.csv",
            mime="text/csv",
        )