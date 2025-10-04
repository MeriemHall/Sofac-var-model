import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import warnings
import base64
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SOFAC - Prédiction VAR Rendements 52-Semaines",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sofac_logo_svg():
    return '''
    <svg width="180" height="60" viewBox="0 0 180 60" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="6" fill="#FFD700"/>
        <path d="M12 28 Q24 20 36 28 Q48 36 60 28 Q72 20 84 28" 
              stroke="#1e3c72" stroke-width="3" fill="none"/>
        <text x="12" y="45" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#1e3c72">SOFAC</text>
        <text x="12" y="57" font-family="Arial, sans-serif" font-size="8" fill="#FF6B35">Dites oui au super crédit</text>
    </svg>
    '''

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }}
    .executive-dashboard {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }}
    .status-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }}
    .metric-box {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-top: 3px solid #2a5298;
    }}
    .recommendation-panel {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }}
    .stMetric label {{ font-size: 0.75rem !important; }}
    h1 {{ font-size: 1.4rem !important; }}
    h2 {{ font-size: 1.2rem !important; }}
    p {{ font-size: 0.82rem !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load CSV data and prepare for VAR modeling"""
    try:
        # Try to load the CSV file
        df = pd.read_csv('sofac csv.csv', sep=';', parse_dates=['Date (Monthly)'])
        
        # Rename columns
        cols = {
            'Date (Monthly)': 'Date',
            'Taux Directeur (%)2': 'Policy_Rate',
            'Inflation sous-jacente (%)': 'Inflation',
            '52 semaines': 'Treasury_Yield'
        }
        df = df.rename(columns=cols)
        
        # Set date as index and sort
        df = df.set_index('Date').sort_index()
        
        # Convert columns to numeric
        for col in ['Policy_Rate', 'Inflation', 'Treasury_Yield']:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Interpolate missing values
        df = df.interpolate()
        
        return df
    except FileNotFoundError:
        # If file not found, create sample data matching the structure
        st.warning("CSV file not found. Using sample data for demonstration.")
        dates = pd.date_range(start='2020-01-01', end='2025-06-01', freq='M')
        
        # Generate realistic sample data
        np.random.seed(42)
        n = len(dates)
        
        policy_rate = np.concatenate([
            np.linspace(2.0, 1.5, 12),  # 2020
            np.full(12, 1.5),            # 2021
            np.linspace(1.5, 3.0, 12),  # 2022
            np.full(12, 3.0),            # 2023
            np.linspace(3.0, 2.5, 12),  # 2024
            np.linspace(2.5, 2.25, 6)   # 2025
        ])[:n]
        
        inflation = np.concatenate([
            np.linspace(0.8, 0.3, 12),
            np.linspace(0.6, 3.6, 12),
            np.linspace(4.8, 7.4, 12),
            np.linspace(7.9, 4.4, 12),
            np.linspace(2.1, 2.3, 12),
            np.linspace(1.4, 1.3, 6)
        ])[:n]
        
        treasury_yield = policy_rate + np.random.normal(0, 0.3, n) + inflation * 0.1
        
        df = pd.DataFrame({
            'Policy_Rate': policy_rate,
            'Inflation': inflation,
            'Treasury_Yield': treasury_yield
        }, index=dates)
        
        return df

@st.cache_data
def train_var_model(df):
    """Train VAR model on differenced data"""
    # Difference the data to make it stationary
    df_diff = df.diff().dropna()
    
    # Fit VAR model
    model = VAR(df_diff)
    
    # Select optimal lag order (max 12 months)
    lag_order = model.select_order(maxlags=12)
    optimal_lag = lag_order.aic
    
    # Fit with optimal lags
    results = model.fit(maxlags=optimal_lag, ic='aic')
    
    return results, df_diff, optimal_lag

@st.cache_data
def generate_var_forecast(_results, df_diff, df_original, forecast_months=60):
    """Generate VAR forecast"""
    # Get initial values for forecasting
    lag_order = _results.k_ar
    initial_values = df_diff.values[-lag_order:]
    
    # Generate forecast
    forecast = _results.forecast(y=initial_values, steps=forecast_months)
    
    # Create future dates
    last_date = df_original.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='M'
    )
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame(
        forecast,
        columns=df_original.columns,
        index=future_dates
    )
    
    # Convert differenced forecast back to levels
    last_actual = df_original.iloc[-1]
    predicted = forecast_df.cumsum() + last_actual
    
    return predicted, forecast_df

def generate_scenarios(predicted_base, forecast_months=60):
    """Generate optimistic and conservative scenarios"""
    scenarios = {}
    
    # Base case
    scenarios['Cas_de_Base'] = predicted_base.copy()
    
    # Conservative scenario (higher rates)
    conservative = predicted_base.copy()
    conservative['Policy_Rate'] = conservative['Policy_Rate'] + 0.3
    conservative['Treasury_Yield'] = conservative['Treasury_Yield'] + 0.25
    conservative['Inflation'] = conservative['Inflation'] + 0.2
    scenarios['Conservateur'] = conservative
    
    # Optimistic scenario (lower rates)
    optimistic = predicted_base.copy()
    optimistic['Policy_Rate'] = optimistic['Policy_Rate'] - 0.25
    optimistic['Treasury_Yield'] = optimistic['Treasury_Yield'] - 0.2
    optimistic['Inflation'] = optimistic['Inflation'] - 0.15
    scenarios['Optimiste'] = optimistic
    
    return scenarios

def calculate_loan_analysis(scenarios, loan_amount, loan_duration, fixed_rate, risk_premium):
    """Calculate loan cost analysis for each scenario"""
    analysis_results = {}
    
    for scenario_name, scenario_df in scenarios.items():
        # Get relevant years
        years_data = scenario_df.head(loan_duration * 12)
        
        # Calculate annual variable rates
        variable_rates_annual = []
        for year in range(loan_duration):
            start_month = year * 12
            end_month = min((year + 1) * 12, len(years_data))
            
            if end_month <= len(years_data):
                year_data = years_data.iloc[start_month:end_month]
                avg_treasury_yield = year_data['Treasury_Yield'].mean()
                effective_rate = avg_treasury_yield + risk_premium
                variable_rates_annual.append(effective_rate)
        
        # Calculate costs
        fixed_cost_total = (fixed_rate / 100) * loan_amount * 1_000_000 * loan_duration
        variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 
                                   for rate in variable_rates_annual])
        
        cost_difference = variable_cost_total - fixed_cost_total
        
        # Risk metrics
        volatility = years_data['Treasury_Yield'].std()
        max_rate = max(variable_rates_annual) if variable_rates_annual else 0
        min_rate = min(variable_rates_annual) if variable_rates_annual else 0
        
        analysis_results[scenario_name] = {
            'variable_rates_annual': variable_rates_annual,
            'avg_variable_rate': np.mean(variable_rates_annual) if variable_rates_annual else 0,
            'fixed_cost_total': fixed_cost_total,
            'variable_cost_total': variable_cost_total,
            'cost_difference': cost_difference,
            'volatility': volatility,
            'max_rate': max_rate,
            'min_rate': min_rate
        }
    
    return analysis_results

def main():
    # Header
    col_logo, col_text = st.columns([1, 3])
    
    with col_logo:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%); 
                    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <h1 style="margin: 0; color: white;">Modèle VAR - Prédiction des Rendements</h1>
            <p style="margin: 0.5rem 0; color: white;">Analyse Vectorielle Autorégressive 52-Semaines</p>
            <p style="margin: 0; color: white;">Données Bank Al-Maghrib | Méthodologie VAR Avancée</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data and train model
    if 'data_loaded' not in st.session_state:
        with st.spinner("Chargement des données et calibration du modèle VAR..."):
            st.session_state.df = load_and_prepare_data()
            st.session_state.var_results, st.session_state.df_diff, st.session_state.optimal_lag = train_var_model(st.session_state.df)
            st.session_state.predicted, st.session_state.forecast_diff = generate_var_forecast(
                st.session_state.var_results,
                st.session_state.df_diff,
                st.session_state.df
            )
            st.session_state.scenarios = generate_scenarios(st.session_state.predicted)
            st.session_state.data_loaded = True
    
    # Sidebar
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Informations du Modèle")
        
        st.markdown("### Configuration VAR")
        st.metric("Ordre de Retard Optimal", f"{st.session_state.optimal_lag} mois")
        st.metric("Période Historique", f"{len(st.session_state.df)} mois")
        st.metric("Horizon de Prédiction", "60 mois (5 ans)")
        
        # Current values
        st.markdown("### Valeurs Actuelles")
        last_values = st.session_state.df.iloc[-1]
        st.metric("Taux Directeur", f"{last_values['Policy_Rate']:.2f}%")
        st.metric("Inflation", f"{last_values['Inflation']:.2f}%")
        st.metric("Rendement 52s", f"{last_values['Treasury_Yield']:.2f}%")
        
        # Strategic outlook
        st.markdown("---")
        st.subheader("🎯 Vision Stratégique")
        
        cas_base = st.session_state.scenarios['Cas_de_Base']
        three_month_avg = cas_base.head(3)['Treasury_Yield'].mean()
        current_yield = last_values['Treasury_Yield']
        trend = "↗️ Hausse" if three_month_avg > current_yield else "↘️ Baisse"
        
        st.metric("Tendance 3 mois", f"{three_month_avg:.2f}%", delta=trend)
        
        six_month_data = cas_base.head(6)
        st.metric("Fourchette 6 mois",
                 f"{six_month_data['Treasury_Yield'].min():.2f}%-{six_month_data['Treasury_Yield'].max():.2f}%")
        
        if st.sidebar.button("Actualiser"):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Vue d'Ensemble", "Prédictions Détaillées", "Analyse Décisionnelle"])
    
    with tab1:
        st.markdown('<div class="executive-dashboard">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 2rem;">Tableau de Bord Stratégique VAR</div>', unsafe_allow_html=True)
        
        # Strategic metrics
        cas_base = st.session_state.scenarios['Cas_de_Base']
        q1_avg = cas_base.head(3)['Treasury_Yield'].mean()
        q2_avg = cas_base.head(6)['Treasury_Yield'].mean()
        year1_avg = cas_base.head(12)['Treasury_Yield'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">HORIZON 3 MOIS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{q1_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">HORIZON 6 MOIS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{q2_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">HORIZON 12 MOIS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{year1_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volatility = cas_base.head(12)['Treasury_Yield'].std()
            risk_level = "Faible" if volatility < 0.3 else "Modéré" if volatility < 0.6 else "Élevé"
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">VOLATILITÉ</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chart
        st.subheader("Évolution des Rendements - Modèle VAR")
        
        fig = go.Figure()
        
        # Historical data
        df_hist = st.session_state.df.tail(12)
        fig.add_trace(go.Scatter(
            x=df_hist.index,
            y=df_hist['Treasury_Yield'],
            mode='lines+markers',
            name='Historique',
            line=dict(color='#2a5298', width=4),
            marker=dict(size=8)
        ))
        
        # Predictions
        colors = {'Conservateur': '#dc3545', 'Cas_de_Base': '#17a2b8', 'Optimiste': '#28a745'}
        for scenario_name, scenario_df in st.session_state.scenarios.items():
            sample_data = scenario_df[::3]  # Sample every 3 months
            fig.add_trace(go.Scatter(
                x=sample_data.index,
                y=sample_data['Treasury_Yield'],
                mode='lines+markers',
                name=scenario_name,
                line=dict(color=colors[scenario_name], width=3),
                marker=dict(size=5)
            ))
        
        current_yield = st.session_state.df.iloc[-1]['Treasury_Yield']
        fig.add_hline(y=current_yield, line_dash="dash", line_color="gray",
                     annotation_text=f"Actuel: {current_yield:.2f}%")
        
        fig.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Période",
            yaxis_title="Rendement 52-Semaines (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Prédictions Détaillées VAR")
        
        scenario_choice = st.selectbox("Choisissez un scénario:",
                                      ['Cas_de_Base', 'Conservateur', 'Optimiste'])
        
        pred_data = st.session_state.scenarios[scenario_choice]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rendement Moyen", f"{pred_data['Treasury_Yield'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_data['Treasury_Yield'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_data['Treasury_Yield'].max():.2f}%")
        
        # Detailed multivariate chart
        st.subheader(f"Prédictions Multivariées - {scenario_choice}")
        
        fig_detail = go.Figure()
        
        # Sample data for clarity
        sample_data = pred_data[::3]
        
        fig_detail.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['Treasury_Yield'],
            name='Rendement 52s',
            line=dict(color='#2a5298', width=3)
        ))
        
        fig_detail.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['Policy_Rate'],
            name='Taux Directeur',
            line=dict(color='#dc3545', width=2, dash='dash')
        ))
        
        fig_detail.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['Inflation'],
            name='Inflation',
            line=dict(color='#28a745', width=2, dash='dot')
        ))
        
        fig_detail.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Valeur (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Export
        if st.button("Télécharger les Prédictions"):
            csv = pred_data.to_csv()
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_var_predictions_{scenario_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("Analyse Décisionnelle")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🏦 AIDE À LA DÉCISION EMPRUNT SOFAC</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse comparative Taux Fixe vs Taux Variable (Modèle VAR)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Loan parameters
        st.subheader("⚙️ Paramètres de l'Emprunt")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 5, 5)
        with col3:
            fixed_rate = st.number_input("Taux fixe (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_premium = st.number_input("Prime de risque (%):", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
        
        # Calculate analysis
        analysis = calculate_loan_analysis(
            st.session_state.scenarios,
            loan_amount,
            loan_duration,
            fixed_rate,
            risk_premium
        )
        
        # Decision matrix
        st.subheader("📊 Matrice de Décision VAR")
        
        decision_data = []
        for scenario_name, result in analysis.items():
            if result['cost_difference'] < 0:
                recommendation = "TAUX VARIABLE"
                savings = abs(result['cost_difference'])
                decision_text = f"Économie: {savings:,.0f} MAD"
                decision_color = "#28a745"
            else:
                recommendation = "TAUX FIXE"
                extra_cost = result['cost_difference']
                decision_text = f"Surcoût: {extra_cost:,.0f} MAD"
                decision_color = "#dc3545"
            
            decision_data.append({
                'Scénario': scenario_name,
                'Taux Variable Moyen': f"{result['avg_variable_rate']:.2f}%",
                'Fourchette': f"{result['min_rate']:.2f}%-{result['max_rate']:.2f}%",
                'Coût Total Variable': f"{result['variable_cost_total']:,.0f} MAD",
                'Différence vs Fixe': decision_text,
                'Recommandation': recommendation
            })
        
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        # Final recommendation
        variable_count = sum(1 for r in analysis.values() if r['cost_difference'] < 0)
        avg_diff = np.mean([r['cost_difference'] for r in analysis.values()])
        
        if variable_count >= 2 and avg_diff < -200000:
            final_rec = "TAUX VARIABLE"
            final_color = "#28a745"
            final_reason = "Majorité des scénarios VAR favorisent le taux variable"
        elif variable_count == 0:
            final_rec = "TAUX FIXE"
            final_color = "#dc3545"
            final_reason = "Tous les scénarios VAR favorisent le taux fixe"
        else:
            final_rec = "STRATÉGIE MIXTE"
            final_color = "#ffc107"
            final_reason = "Signaux mixtes - approche équilibrée recommandée"
        
        st.markdown(f"""
        <div class="recommendation-panel" style="background: linear-gradient(135deg, {final_color}, {final_color}AA);">
            <h2>🎯 DÉCISION FINALE SOFAC (VAR)</h2>
            <h3>{final_rec}</h3>
            <p><strong>Justification:</strong> {final_reason}</p>
            <p><strong>Montant:</strong> {loan_amount}M MAD | <strong>Durée:</strong> {loan_duration} ans</p>
            <p><strong>Méthodologie:</strong> Vector Autoregression (VAR) - Ordre {st.session_state.optimal_lag}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Modèle VAR de Prédiction des Rendements</p>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Méthodologie VAR Avancée | Dernière mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Prédictions basées sur l'analyse vectorielle autorégressive</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

