import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SOFAC - Outil de Planification",
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

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .executive-dashboard {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-top: 3px solid #2a5298;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .validation-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stMetric label { font-size: 0.75rem !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    p { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_scenarios():
    """Load scenarios from Excel file"""
    try:
        # Try to load the Excel file with your scenarios
        base = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Base')
        optimistic = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Optimistic')
        conservative = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Conservative')
        
        # Ensure Date column is datetime
        for df in [base, optimistic, conservative]:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return {
            'Cas_de_Base': base,
            'Optimiste': optimistic,
            'Conservateur': conservative
        }
    except FileNotFoundError:
        st.error("Fichier Excel 'SOFAC_Scenarios_24Months.xlsx' non trouvé. Veuillez le placer dans le même dossier que app.py")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des scénarios: {str(e)}")
        return None

@st.cache_data
def load_historical_data():
    """Load historical data for validation"""
    try:
        # Load your CSV with historical data
        df = pd.read_csv('sofac csv.csv', sep=';')
        
        # Rename columns
        df = df.rename(columns={
            'Date (Monthly)': 'Date',
            'Taux Directeur (%)2': 'Policy_Rate',
            'Inflation sous-jacente (%)': 'Core_Inflation',
            '52 semaines': 'Treasury_Yield'
        })
        
        # Convert to proper types
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        for col in ['Policy_Rate', 'Core_Inflation', 'Treasury_Yield']:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        st.warning("Fichier historique non trouvé. Utilisation de données limitées.")
        return None
    except Exception as e:
        st.warning(f"Erreur de chargement historique: {str(e)}")
        return None

def calculate_model_performance(historical_df):
    """Calculate model performance metrics"""
    if historical_df is None or len(historical_df) < 12:
        return {
            'mae': 0.218,  # Your validated results
            'mape': 10.7,
            'rmse': 0.28,
            'test_period': '2025'
        }
    
    # Use last 12 months for testing
    test_df = historical_df.tail(12).copy()
    
    # Calculate predicted yields using your regression formula
    test_df['Predicted_Yield'] = (0.024 + 
                                   (0.9998 * test_df['Policy_Rate']) + 
                                   (0.0444 * test_df['Core_Inflation']))
    
    # Calculate errors
    test_df['Error'] = abs(test_df['Treasury_Yield'] - test_df['Predicted_Yield'])
    test_df['Pct_Error'] = abs((test_df['Treasury_Yield'] - test_df['Predicted_Yield']) / 
                                test_df['Treasury_Yield']) * 100
    
    mae = test_df['Error'].mean()
    mape = test_df['Pct_Error'].mean()
    rmse = np.sqrt(((test_df['Treasury_Yield'] - test_df['Predicted_Yield'])**2).mean())
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'test_period': f"{test_df['Date'].min().strftime('%b %Y')} - {test_df['Date'].max().strftime('%b %Y')}",
        'test_data': test_df
    }

def calculate_loan_analysis(scenarios, loan_amount, loan_duration, fixed_rate, risk_premium):
    """Calculate loan cost analysis for each scenario"""
    analysis_results = {}
    
    max_duration = min(loan_duration, 2)  # Limit to 2 years (24 months)
    
    for scenario_name, scenario_df in scenarios.items():
        years_data = scenario_df.head(max_duration * 12)
        
        variable_rates_annual = []
        for year in range(max_duration):
            start_month = year * 12
            end_month = min((year + 1) * 12, len(years_data))
            
            if end_month <= len(years_data):
                year_data = years_data.iloc[start_month:end_month]
                avg_treasury_yield = year_data['Predicted_Yield'].mean()
                effective_rate = avg_treasury_yield + risk_premium
                variable_rates_annual.append(effective_rate)
        
        fixed_cost_total = (fixed_rate / 100) * loan_amount * 1_000_000 * max_duration
        variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 
                                   for rate in variable_rates_annual])
        
        cost_difference = variable_cost_total - fixed_cost_total
        
        volatility = years_data['Predicted_Yield'].std()
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
            <h1 style="margin: 0; color: white;">Outil de Planification de Scénarios</h1>
            <p style="margin: 0.5rem 0; color: white;">Analyse de Sensibilité Taux Fixe vs Variable</p>
            <p style="margin: 0; color: white;">Basé sur Modèle de Régression - Horizon 24 Mois</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    if 'data_loaded' not in st.session_state:
        with st.spinner("Chargement des scénarios..."):
            st.session_state.scenarios = load_scenarios()
            st.session_state.historical = load_historical_data()
            st.session_state.performance = calculate_model_performance(st.session_state.historical)
            st.session_state.data_loaded = True
    
    if st.session_state.scenarios is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Performance du Modèle")
        
        perf = st.session_state.performance
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Précision du Modèle")
        st.metric("Erreur Moyenne (MAE)", f"±{perf['mae']:.2f}%")
        st.metric("Erreur Relative (MAPE)", f"{perf['mape']:.1f}%")
        st.caption(f"Testé sur: {perf['test_period']}")
        st.markdown("""
        **Interprétation:** Précision acceptable pour analyse comparative de scénarios.
        Ne pas utiliser pour prédictions exactes.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Configuration")
        st.metric("Modèle", "Régression Linéaire")
        st.metric("R² (entraînement)", "94.3%")
        st.metric("Horizon", "24 mois")
        
        # Current values
        if st.session_state.historical is not None:
            last_values = st.session_state.historical.iloc[-1]
            st.markdown("### Valeurs Actuelles")
            st.metric("Taux Directeur", f"{last_values['Policy_Rate']:.2f}%")
            st.metric("Inflation", f"{last_values['Core_Inflation']:.2f}%")
            st.metric("Rendement 52s", f"{last_values['Treasury_Yield']:.2f}%")
        
        if st.sidebar.button("Actualiser"):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Analyse de Scénarios", "Comparaison Détaillée", "Analyse Décisionnelle"])
    
    with tab1:
        st.header("Planification par Scénarios")
        
        st.markdown("""
        <div class="warning-box">
            <h4>📋 Objectif de cet Outil</h4>
            <p>Cet outil génère des <strong>scénarios exploratoires</strong> basés sur les orientations de Bank Al-Maghrib 
            pour vous aider à <strong>comparer différentes options de financement</strong> (taux fixe vs variable).</p>
            <p><strong>Important:</strong> Les valeurs exactes ne sont pas des prédictions fiables (erreur moyenne: ±{:.2f}%). 
            L'intérêt est dans la <strong>comparaison relative</strong> entre scénarios.</p>
        </div>
        """.format(st.session_state.performance['mae']), unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown(f"""
        <div class="status-card">
            <h4>Performance Historique du Modèle</h4>
            <p><strong>Période de test:</strong> {st.session_state.performance['test_period']}</p>
            <p><strong>Erreur moyenne:</strong> ±{st.session_state.performance['mae']:.2f}% | 
            <strong>Erreur relative:</strong> {st.session_state.performance['mape']:.1f}%</p>
            <p>Le modèle est utilisé pour <strong>l'analyse comparative</strong>, pas pour prédire des valeurs précises.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation chart if available
        if 'test_data' in st.session_state.performance:
            st.subheader("Test Rétrospectif du Modèle")
            
            test_df = st.session_state.performance['test_data']
            
            fig_val = go.Figure()
            
            fig_val.add_trace(go.Scatter(
                x=test_df['Date'],
                y=test_df['Treasury_Yield'],
                mode='lines+markers',
                name='Valeurs Réelles',
                line=dict(color='#2a5298', width=3),
                marker=dict(size=8)
            ))
            
            fig_val.add_trace(go.Scatter(
                x=test_df['Date'],
                y=test_df['Predicted_Yield'],
                mode='lines+markers',
                name='Estimations du Modèle',
                line=dict(color='#dc3545', width=3, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig_val.update_layout(
                title="Comparaison Historique: Estimations vs Réalité",
                height=400,
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Rendement 52-Semaines (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_val, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
                <p><strong>Interprétation:</strong> L'écart entre les courbes montre que le modèle capture les tendances générales 
                mais n'est pas précis pour des valeurs exactes. Utilisez-le pour comprendre les <strong>directions possibles</strong> 
                et <strong>comparer des stratégies</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Scénarios Exploratoires - Horizon 24 Mois")
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Ces scénarios ne sont pas des prédictions:</strong> Ils représentent des 
            trajectoires <strong>hypothétiques</strong> basées sur les orientations de BAM. L'objectif est 
            de vous aider à <strong>tester différentes hypothèses</strong> et évaluer la sensibilité de vos 
            décisions de financement.
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        cas_base = st.session_state.scenarios['Cas_de_Base']
        optimiste = st.session_state.scenarios['Optimiste']
        conservateur = st.session_state.scenarios['Conservateur']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Scénario Médian (24m)", f"{cas_base['Predicted_Yield'].mean():.2f}%")
        with col2:
            st.metric("Fourchette des Scénarios", 
                     f"{optimiste['Predicted_Yield'].mean():.2f}% - {conservateur['Predicted_Yield'].mean():.2f}%")
        with col3:
            if st.session_state.historical is not None:
                current = st.session_state.historical.iloc[-1]['Treasury_Yield']
                st.metric("Point de Départ", f"{current:.2f}%", help="Dernière valeur observée")
        
        # Main comparison chart
        fig = go.Figure()
        
        # Historical data if available
        if st.session_state.historical is not None:
            df_hist = st.session_state.historical.tail(12)
            fig.add_trace(go.Scatter(
                x=df_hist['Date'],
                y=df_hist['Treasury_Yield'],
                mode='lines+markers',
                name='Données Historiques',
                line=dict(color='#2a5298', width=4),
                marker=dict(size=8)
            ))
        
        # Scenarios with uncertainty bands
        colors = {'Conservateur': '#dc3545', 'Cas_de_Base': '#17a2b8', 'Optimiste': '#28a745'}
        labels = {
            'Conservateur': 'Scénario Haussier (Conservateur)',
            'Cas_de_Base': 'Scénario Médian (Cas de Base)',
            'Optimiste': 'Scénario Baissier (Optimiste)'
        }
        
        for scenario_name, scenario_df in st.session_state.scenarios.items():
            # Main line
            fig.add_trace(go.Scatter(
                x=scenario_df['Date'],
                y=scenario_df['Predicted_Yield'],
                mode='lines+markers',
                name=labels[scenario_name],
                line=dict(color=colors[scenario_name], width=3),
                marker=dict(size=5)
            ))
            
            # Uncertainty band for base case only
            if scenario_name == 'Cas_de_Base':
                fig.add_trace(go.Scatter(
                    x=scenario_df['Date'],
                    y=scenario_df['Upper_Bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=scenario_df['Date'],
                    y=scenario_df['Lower_Bound'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(23, 162, 184, 0.2)',
                    line=dict(width=0),
                    name='Bande d\'incertitude (±0.22%)',
                    hoverinfo='skip'
                ))
        
        if st.session_state.historical is not None:
            current_yield = st.session_state.historical.iloc[-1]['Treasury_Yield']
            fig.add_hline(y=current_yield, line_dash="dot", line_color="gray",
                         annotation_text=f"Actuel: {current_yield:.2f}%")
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement 52-Semaines (%) - Scénarios Hypothétiques",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario interpretation guide
        st.subheader("Comment Interpréter ces Scénarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #28a745;">
                <h4 style="color: #28a745;">Scénario Baissier (Optimiste)</h4>
                <p style="font-size: 0.85rem;"><strong>Moyenne:</strong> {optimiste['Predicted_Yield'].mean():.2f}%</p>
                <p style="font-size: 0.85rem;">Hypothèse d'assouplissement monétaire renforcé. 
                Utiliser pour tester: "Et si les taux baissent plus que prévu?"</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #17a2b8;">
                <h4 style="color: #17a2b8;">Scénario Médian (Base)</h4>
                <p style="font-size: 0.85rem;"><strong>Moyenne:</strong> {cas_base['Predicted_Yield'].mean():.2f}%</p>
                <p style="font-size: 0.85rem;">Basé sur les orientations BAM Q3 2025. 
                Scénario de référence pour comparaisons.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #dc3545;">
                <h4 style="color: #dc3545;">Scénario Haussier (Conservateur)</h4>
                <p style="font-size: 0.85rem;"><strong>Moyenne:</strong> {conservateur['Predicted_Yield'].mean():.2f}%</p>
                <p style="font-size: 0.85rem;">Hypothèse de tensions inflationnistes. 
                Utiliser pour tester: "Et si les taux remontent?"</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show scenario details
        with st.expander("Voir les détails des scénarios"):
            scenario_choice = st.selectbox("Choisir un scénario:", 
                                          ['Cas_de_Base', 'Optimiste', 'Conservateur'])
            selected_df = st.session_state.scenarios[scenario_choice]
            
            st.dataframe(
                selected_df[['Date', 'Policy_Rate', 'Core_Inflation', 'Predicted_Yield', 
                            'Lower_Bound', 'Upper_Bound']].head(24),
                use_container_width=True
            )
            
            # Download option
            csv = selected_df.to_csv(index=False)
            st.download_button(
                label=f"Télécharger {scenario_choice} (CSV)",
                data=csv,
                file_name=f"sofac_scenario_{scenario_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("Analyse Comparative Taux Fixe vs Variable")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">📊 Outil d'Analyse de Sensibilité</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Comparez l'impact de différents scénarios économiques sur vos options de financement</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Paramètres de l'Emprunt")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 5, 2)
        with col3:
            fixed_rate = st.number_input("Taux fixe proposé (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_premium = st.number_input("Prime de risque variable (%):", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
        
        # Warning for long duration
        if loan_duration > 2:
            st.markdown("""
            <div class="warning-box">
                <strong>Note:</strong> L'analyse est limitée aux 2 premières années en raison de l'incertitude 
                croissante des scénarios au-delà de cet horizon.
            </div>
            """, unsafe_allow_html=True)
        
        analysis = calculate_loan_analysis(
            st.session_state.scenarios,
            loan_amount,
            loan_duration,
            fixed_rate,
            risk_premium
        )
        
        # Comparative analysis table
        st.subheader("Comparaison par Scénario")
        
        decision_data = []
        for scenario_name, result in analysis.items():
            scenario_labels = {
                'Conservateur': 'Haussier (taux ↗)',
                'Cas_de_Base': 'Médian (tendance)',
                'Optimiste': 'Baissier (taux ↘)'
            }
            
            if result['cost_difference'] < 0:
                recommendation = "Variable avantageux"
                decision_text = f"Économie: {abs(result['cost_difference']):,.0f} MAD"
                color_indicator = "🟢"
            else:
                recommendation = "Fixe avantageux"
                decision_text = f"Surcoût variable: {result['cost_difference']:,.0f} MAD"
                color_indicator = "🔴"
            
            decision_data.append({
                'Scénario': f"{color_indicator} {scenario_labels[scenario_name]}",
                'Taux Variable Moyen': f"{result['avg_variable_rate']:.2f}%",
                'vs Fixe ({:.2f}%)'.format(fixed_rate): decision_text,
                'Dans ce scénario': recommendation
            })
        
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        variable_count = sum(1 for r in analysis.values() if r['cost_difference'] < 0)
        
        st.subheader("💡 Interprétation de l'Analyse")
        
        if variable_count == 3:
            interpretation = """
            **Tous les scénarios favorisent le taux variable**
            
            Même dans un scénario haussier (taux en hausse), le taux variable reste avantageux. 
            Cela suggère que le taux fixe proposé ({:.2f}%) est relativement élevé par rapport 
            aux projections du marché.
            
            **Sensibilité:** Faible - La décision est robuste aux différents scénarios.
            """.format(fixed_rate)
            box_color = "#28a745"
            
        elif variable_count == 1:
            interpretation = """
            **Les scénarios sont partagés**
            
            Seul le scénario baissier favorise le taux variable. Dans les scénarios médian et haussier, 
            le taux fixe est plus avantageux.
            
            **Sensibilité:** Élevée - Votre choix doit refléter votre tolérance au risque et votre 
            vision de l'évolution des taux.
            """
            box_color = "#ffc107"
            
        else:  # variable_count == 0
            interpretation = """
            **Tous les scénarios favorisent le taux fixe**
            
            Même dans un scénario baissier (taux en baisse), le taux fixe proposé ({:.2f}%) reste 
            compétitif par rapport aux projections de taux variable.
            
            **Sensibilité:** Faible - Le taux fixe semble attractif dans tous les cas envisagés.
            """.format(fixed_rate)
            box_color = "#dc3545"
        
        st.markdown(f"""
        <div style="background: {box_color}22; border-left: 4px solid {box_color}; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            {interpretation}
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed yearly breakdown
        st.subheader("📊 Détail Annuel des Coûts")
        
        for scenario_name, result in analysis.items():
            with st.expander(f"Scénario: {scenario_name}"):
                if result['variable_rates_annual']:
                    years_df = pd.DataFrame({
                        'Année': [f"Année {i+1}" for i in range(len(result['variable_rates_annual']))],
                        'Taux Variable Effectif': [f"{rate:.2f}%" for rate in result['variable_rates_annual']],
                        'Taux Fixe': [f"{fixed_rate:.2f}%"] * len(result['variable_rates_annual']),
                        'Coût Variable': [f"{(rate/100) * loan_amount * 1_000_000:,.0f} MAD" 
                                         for rate in result['variable_rates_annual']],
                        'Coût Fixe': [f"{(fixed_rate/100) * loan_amount * 1_000_000:,.0f} MAD"] * len(result['variable_rates_annual'])
                    })
                    st.dataframe(years_df, use_container_width=True, hide_index=True)
                    
                    st.markdown(f"""
                    **Résumé sur {result['analysis_period']} an(s):**
                    - Coût total variable: {result['variable_cost_total']:,.0f} MAD
                    - Coût total fixe: {result['fixed_cost_total']:,.0f} MAD
                    - Différence: {result['cost_difference']:+,.0f} MAD
                    """)
        
        # Key message
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Points Importants</h4>
            <ul>
                <li>Cette analyse est basée sur des <strong>scénarios hypothétiques</strong>, pas des prédictions</li>
                <li>Les taux réels futurs peuvent différer significativement de ces scénarios</li>
                <li>Utilisez cette analyse pour <strong>comprendre votre sensibilité</strong> aux variations de taux</li>
                <li>Considérez d'autres facteurs: situation financière, tolérance au risque, flexibilité de remboursement</li>
                <li>Consultez votre conseiller SOFAC pour une analyse personnalisée complète</li>
            </ul>
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
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Outil de Planification de Scénarios</p>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Modèle de Régression (R²=94.3%, MAE=±{st.session_state.performance['mae']:.2f}%) | Mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Outil d'aide à la décision - Les scénarios ne sont pas des prédictions</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 2:
            interpretation = """
            **La majorité des scénarios favorisent le taux variable**
            
            Le taux variable est avantageux dans les scénarios médian et baissier, mais pourrait 
            coûter plus cher si les taux montent significativement (scénario haussier).
            
            **Sensibilité:** Modérée - La décision dépend de votre anticipation du cycle économique.
            """
            box_color = "#17a2b8"
            
        elif variable_count ==
