import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
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
    .validation-box {{
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    .warning-box {{
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
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
        df = pd.read_csv('sofac csv.csv', sep=';', parse_dates=['Date (Monthly)'])
        
        cols = {
            'Date (Monthly)': 'Date',
            'Taux Directeur (%)2': 'Policy_Rate',
            'Inflation sous-jacente (%)': 'Inflation',
            '52 semaines': 'Treasury_Yield'
        }
        df = df.rename(columns=cols)
        df = df.set_index('Date').sort_index()
        
        for col in ['Policy_Rate', 'Inflation', 'Treasury_Yield']:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        df = df.interpolate()
        return df
        
    except FileNotFoundError:
        st.warning("CSV file not found. Using sample data for demonstration.")
        dates = pd.date_range(start='2018-03-01', end='2025-09-01', freq='M')
        
        np.random.seed(42)
        n = len(dates)
        
        policy_rate = np.concatenate([
            np.linspace(2.25, 1.5, 28),
            np.full(12, 1.5),
            np.linspace(1.5, 3.0, 12),
            np.full(12, 3.0),
            np.linspace(3.0, 2.25, 26)
        ])[:n]
        
        inflation = np.concatenate([
            np.linspace(1.0, 0.8, 12),
            np.linspace(0.8, 7.9, 24),
            np.linspace(7.9, 4.4, 12),
            np.linspace(4.4, 1.3, 42)
        ])[:n]
        
        treasury_yield = policy_rate + np.random.normal(0, 0.2, n) + inflation * 0.05
        
        df = pd.DataFrame({
            'Policy_Rate': policy_rate,
            'Inflation': inflation,
            'Treasury_Yield': treasury_yield
        }, index=dates)
        
        return df

@st.cache_data
def validate_model(df, test_months=12):
    """Validate model on recent data"""
    train_df = df.iloc[:-test_months]
    test_df = df.iloc[-test_months:]
    
    # Train on historical data
    train_diff = train_df.diff().dropna()
    model = VAR(train_diff)
    results = model.fit(maxlags=12, ic='aic')
    
    # Forecast test period
    lag_order = results.k_ar
    initial_values = train_diff.values[-lag_order:]
    forecast = results.forecast(y=initial_values, steps=test_months)
    
    # Convert back to levels
    forecast_df = pd.DataFrame(forecast, columns=train_df.columns, index=test_df.index)
    predicted_levels = forecast_df.cumsum() + train_df.iloc[-1]
    
    # Calculate errors
    mae = mean_absolute_error(test_df['Treasury_Yield'], predicted_levels['Treasury_Yield'])
    rmse = np.sqrt(mean_squared_error(test_df['Treasury_Yield'], predicted_levels['Treasury_Yield']))
    mape = np.mean(np.abs((test_df['Treasury_Yield'] - predicted_levels['Treasury_Yield']) / test_df['Treasury_Yield'])) * 100
    
    return {
        'actual': test_df,
        'predicted': predicted_levels,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'train_end_date': train_df.index[-1],
        'test_start_date': test_df.index[0]
    }

@st.cache_data
def train_var_model(df):
    """Train VAR model on full dataset"""
    df_diff = df.diff().dropna()
    model = VAR(df_diff)
    lag_order = model.select_order(maxlags=12)
    optimal_lag = lag_order.aic
    results = model.fit(maxlags=optimal_lag, ic='aic')
    
    return results, df_diff, optimal_lag

@st.cache_data
def generate_var_forecast(_results, df_diff, df_original, forecast_months=24):
    """Generate VAR forecast for 24 months (2 years)"""
    lag_order = _results.k_ar
    initial_values = df_diff.values[-lag_order:]
    
    # Generate point forecast
    forecast = _results.forecast(y=initial_values, steps=forecast_months)
    
    last_date = df_original.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='M'
    )
    
    forecast_df = pd.DataFrame(forecast, columns=df_original.columns, index=future_dates)
    last_actual = df_original.iloc[-1]
    predicted = forecast_df.cumsum() + last_actual
    
    return predicted, forecast_df

def generate_scenarios_with_uncertainty(base_prediction, validation_mae):
    """Generate scenarios with uncertainty bands based on validation"""
    scenarios = {}
    
    # Base case - the point forecast
    scenarios['Cas_de_Base'] = base_prediction.copy()
    
    # Conservative scenario - upper bound (mean + 1.5 * MAE)
    conservative = base_prediction.copy()
    conservative['Treasury_Yield'] = conservative['Treasury_Yield'] + 1.5 * validation_mae
    conservative['Policy_Rate'] = conservative['Policy_Rate'] + 1.2 * validation_mae
    conservative['Inflation'] = conservative['Inflation'] + 0.3
    scenarios['Conservateur'] = conservative
    
    # Optimistic scenario - lower bound (mean - 1.5 * MAE)
    optimistic = base_prediction.copy()
    optimistic['Treasury_Yield'] = optimistic['Treasury_Yield'] - 1.5 * validation_mae
    optimistic['Policy_Rate'] = optimistic['Policy_Rate'] - 1.2 * validation_mae
    optimistic['Inflation'] = optimistic['Inflation'] - 0.3
    scenarios['Optimiste'] = optimistic
    
    # Apply realistic bounds
    for scenario in scenarios.values():
        scenario['Treasury_Yield'] = np.clip(scenario['Treasury_Yield'], 0.5, 5.0)
        scenario['Policy_Rate'] = np.clip(scenario['Policy_Rate'], 0.5, 4.5)
        scenario['Inflation'] = np.clip(scenario['Inflation'], 0.2, 4.0)
    
    return scenarios

def calculate_loan_analysis(scenarios, loan_amount, loan_duration, fixed_rate, risk_premium):
    """Calculate loan cost analysis - adjusted for 2-year max horizon"""
    analysis_results = {}
    
    # Limit loan duration to available prediction horizon
    max_duration = min(loan_duration, 2)  # 2 years max for predictions
    
    for scenario_name, scenario_df in scenarios.items():
        years_data = scenario_df.head(max_duration * 12)
        
        variable_rates_annual = []
        for year in range(max_duration):
            start_month = year * 12
            end_month = min((year + 1) * 12, len(years_data))
            
            if end_month <= len(years_data):
                year_data = years_data.iloc[start_month:end_month]
                avg_treasury_yield = year_data['Treasury_Yield'].mean()
                effective_rate = avg_treasury_yield + risk_premium
                variable_rates_annual.append(effective_rate)
        
        # Calculate costs only for available period
        fixed_cost_total = (fixed_rate / 100) * loan_amount * 1_000_000 * max_duration
        variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 
                                   for rate in variable_rates_annual])
        
        cost_difference = variable_cost_total - fixed_cost_total
        
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
            'min_rate': min_rate,
            'analysis_period': max_duration
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
            <h1 style="margin: 0; color: white;">Modèle VAR Validé - Prédiction 52-Semaines</h1>
            <p style="margin: 0.5rem 0; color: white;">Analyse Vectorielle Autorégressive avec Validation Historique</p>
            <p style="margin: 0; color: white;">Horizon de Prédiction: 24 Mois (2 Ans)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data and models
    if 'data_loaded' not in st.session_state:
        with st.spinner("Chargement et validation du modèle..."):
            st.session_state.df = load_and_prepare_data()
            st.session_state.validation = validate_model(st.session_state.df, test_months=12)
            st.session_state.var_results, st.session_state.df_diff, st.session_state.optimal_lag = train_var_model(st.session_state.df)
            st.session_state.predicted, st.session_state.forecast_diff = generate_var_forecast(
                st.session_state.var_results,
                st.session_state.df_diff,
                st.session_state.df,
                forecast_months=24
            )
            st.session_state.scenarios = generate_scenarios_with_uncertainty(
                st.session_state.predicted,
                st.session_state.validation['mae']
            )
            st.session_state.data_loaded = True
    
    # Sidebar
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Performance du Modèle")
        
        val = st.session_state.validation
        
        st.markdown('<div class="validation-box">', unsafe_allow_html=True)
        st.markdown("### Validation Historique")
        st.metric("Erreur Moyenne (MAE)", f"±{val['mae']:.3f}%")
        st.metric("RMSE", f"±{val['rmse']:.3f}%")
        st.metric("MAPE", f"{val['mape']:.2f}%")
        st.caption(f"Testé sur {val['test_start_date'].strftime('%b %Y')} - {val['actual'].index[-1].strftime('%b %Y')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Configuration")
        st.metric("Ordre de Retard", f"{st.session_state.optimal_lag} mois")
        st.metric("Données Historiques", f"{len(st.session_state.df)} mois")
        st.metric("Horizon Prédiction", "24 mois")
        
        last_values = st.session_state.df.iloc[-1]
        st.markdown("### Valeurs Actuelles")
        st.metric("Taux Directeur", f"{last_values['Policy_Rate']:.2f}%")
        st.metric("Inflation", f"{last_values['Inflation']:.2f}%")
        st.metric("Rendement 52s", f"{last_values['Treasury_Yield']:.2f}%")
        
        if st.sidebar.button("Actualiser"):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Validation", "Prédictions", "Analyse Décisionnelle"])
    
    with tab1:
        st.header("Validation du Modèle sur Données Historiques")
        
        val = st.session_state.validation
        
        st.markdown(f"""
        <div class="validation-box">
            <h3>Test de Performance Rétrospective</h3>
            <p><strong>Méthode:</strong> Le modèle a été entraîné sur les données jusqu'à {val['train_end_date'].strftime('%B %Y')}, 
            puis a prédit les 12 mois suivants. Les prédictions sont comparées aux valeurs réelles.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Erreur Absolue Moyenne", f"±{val['mae']:.3f}%", 
                     help="En moyenne, les prédictions sont à ±{:.3f}% de la réalité".format(val['mae']))
        with col2:
            st.metric("RMSE", f"±{val['rmse']:.3f}%",
                     help="Erreur quadratique moyenne")
        with col3:
            st.metric("Erreur Relative (MAPE)", f"{val['mape']:.2f}%",
                     help="Erreur en pourcentage de la valeur réelle")
        
        # Validation chart
        fig_val = go.Figure()
        
        fig_val.add_trace(go.Scatter(
            x=val['actual'].index,
            y=val['actual']['Treasury_Yield'],
            mode='lines+markers',
            name='Valeurs Réelles',
            line=dict(color='#2a5298', width=3),
            marker=dict(size=8)
        ))
        
        fig_val.add_trace(go.Scatter(
            x=val['predicted'].index,
            y=val['predicted']['Treasury_Yield'],
            mode='lines+markers',
            name='Prédictions du Modèle',
            line=dict(color='#dc3545', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_val.update_layout(
            title="Comparaison: Prédictions vs Réalité (Période de Test)",
            height=450,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement 52-Semaines (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_val, use_container_width=True)
        
        # Interpretation
        if val['mape'] < 5:
            quality = "EXCELLENT"
            color = "#28a745"
            interpretation = "Le modèle a une très bonne précision pour des prédictions à court terme (12 mois)."
        elif val['mape'] < 10:
            quality = "BON"
            color = "#17a2b8"
            interpretation = "Le modèle offre une précision acceptable pour la planification financière."
        elif val['mape'] < 15:
            quality = "MOYEN"
            color = "#ffc107"
            interpretation = "Le modèle capture les tendances générales mais avec une marge d'erreur notable."
        else:
            quality = "LIMITÉ"
            color = "#dc3545"
            interpretation = "La précision est limitée. Utiliser principalement pour l'analyse de scénarios."
        
        st.markdown(f"""
        <div style="background: {color}22; border-left: 4px solid {color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: {color}; margin: 0;">Niveau de Précision: {quality}</h4>
            <p style="margin: 0.5rem 0 0 0;">{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Prédictions VAR - Horizon 24 Mois")
        
        st.markdown("""
        <div class="warning-box">
            <strong>Important:</strong> Les prédictions sont plus fiables à court terme (3-6 mois) 
            et deviennent plus incertaines au-delà de 12 mois. Les scénarios représentent 
            une fourchette d'incertitude basée sur la performance historique du modèle.
        </div>
        """, unsafe_allow_html=True)
        
        cas_base = st.session_state.scenarios['Cas_de_Base']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rendement Moyen (24m)", f"{cas_base['Treasury_Yield'].mean():.2f}%")
        with col2:
            st.metric("Fourchette", 
                     f"{cas_base['Treasury_Yield'].min():.2f}% - {cas_base['Treasury_Yield'].max():.2f}%")
        with col3:
            current = st.session_state.df.iloc[-1]['Treasury_Yield']
            change = cas_base['Treasury_Yield'].mean() - current
            st.metric("Variation Moyenne", f"{change:+.2f}%")
        
        # Main prediction chart
        fig = go.Figure()
        
        # Historical data
        df_hist = st.session_state.df.tail(24)
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
            fig.add_trace(go.Scatter(
                x=scenario_df.index,
                y=scenario_df['Treasury_Yield'],
                mode='lines+markers',
                name=scenario_name,
                line=dict(color=colors[scenario_name], width=3),
                marker=dict(size=5)
            ))
        
        current_yield = st.session_state.df.iloc[-1]['Treasury_Yield']
        fig.add_hline(y=current_yield, line_dash="dash", line_color="gray",
                     annotation_text=f"Actuel: {current_yield:.2f}%")
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement 52-Semaines (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence intervals
        st.subheader("Intervalles de Confiance")
        st.markdown(f"""
        Les scénarios sont construits à partir de la validation historique:
        - **Conservateur**: Prédiction + 1.5 × MAE ({st.session_state.validation['mae']:.3f}%)
        - **Cas de Base**: Prédiction du modèle VAR
        - **Optimiste**: Prédiction - 1.5 × MAE
        
        Ces intervalles reflètent l'incertitude historique du modèle.
        """)
    
    with tab3:
        st.header("Analyse Décisionnelle")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">Aide à la Décision Emprunt SOFAC</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse sur horizon de prédiction fiable (maximum 2 ans)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Paramètres de l'Emprunt")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 5, 2)
        with col3:
            fixed_rate = st.number_input("Taux fixe (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_premium = st.number_input("Prime de risque (%):", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
        
        # Warning for long duration
        if loan_duration > 2:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Attention:</strong> La durée sélectionnée ({} ans) dépasse l'horizon de prédiction fiable (2 ans). 
                L'analyse sera limitée aux 2 premières années. Pour les années suivantes, l'incertitude est trop élevée 
                pour fournir des prédictions significatives.
            </div>
            """.format(loan_duration), unsafe_allow_html=True)
        
        analysis = calculate_loan_analysis(
            st.session_state.scenarios,
            loan_amount,
            loan_duration,
            fixed_rate,
            risk_premium
        )
        
        # Decision matrix
        st.subheader("Matrice de Décision")
        
        decision_data = []
        for scenario_name, result in analysis.items():
            if result['cost_difference'] < 0:
                recommendation = "TAUX VARIABLE"
                decision_text = f"Économie: {abs(result['cost_difference']):,.0f} MAD"
            else:
                recommendation = "TAUX FIXE"
                decision_text = f"Surcoût: {result['cost_difference']:,.0f} MAD"
            
            decision_data.append({
                'Scénario': scenario_name,
                'Taux Variable Moyen': f"{result['avg_variable_rate']:.2f}%",
                'Fourchette': f"{result['min_rate']:.2f}%-{result['max_rate']:.2f}%",
                'Différence vs Fixe': decision_text,
                'Recommandation': recommendation,
                'Période Analysée': f"{result['analysis_period']} ans"
            })
        
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        # Final recommendation
        variable_count = sum(1 for r in analysis.values() if r['cost_difference'] < 0)
        
        if variable_count >= 2:
            final_rec = "TAUX VARIABLE FAVORABLE"
            final_color = "#28a745"
            final_reason = f"Sur l'horizon de prédiction ({analysis['Cas_de_Base']['analysis_period']} ans), la majorité des scénarios favorisent le taux variable"
        elif variable_count == 0:
            final_rec = "TAUX FIXE RECOMMANDÉ"
            final_color = "#dc3545"
            final_reason = f"Sur l'horizon de prédiction ({analysis['Cas_de_Base']['analysis_period']} ans), tous les scénarios favorisent le taux fixe"
        else:
            final_rec = "SITUATION INCERTAINE"
            final_color = "#ffc107"
            final_reason = "Les scénarios sont partagés - décision à évaluer selon votre tolérance au risque"
        
        st.markdown(f"""
        <div class="recommendation-panel" style="background: linear-gradient(135deg, {final_color}, {final_color}AA);">
            <h2>Recommandation SOFAC</h2>
            <h3>{final_rec}</h3>
            <p><strong>Justification:</strong> {final_reason}</p>
            <p><strong>Horizon d'analyse:</strong> {analysis['Cas_de_Base']['analysis_period']} ans (horizon de prédiction fiable)</p>
            <p><strong>Précision du modèle:</strong> ±{st.session_state.validation['mae']:.3f}% (MAE historique)</p>
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
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Modèle VAR Validé | Horizon 24 Mois</p>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Précision Historique: ±{st.session_state.validation['mae']:.3f}% | Dernière mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Modèle validé sur données historiques - Fiabilité décroissante au-delà de 12 mois</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
