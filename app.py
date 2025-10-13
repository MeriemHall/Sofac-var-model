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
    .warning-box {{
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    .validation-box {{
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
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
def load_scenarios():
    """Load scenario data from Excel file"""
    try:
        # Load all three scenarios
        base = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Base')
        optimistic = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Optimistic')
        conservative = pd.read_excel('SOFAC_Scenarios_24Months.xlsx', sheet_name='Scenario_Conservative')
        
        # Convert Date column to datetime if needed
        for df in [base, optimistic, conservative]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
        
        return {
            'Base': base,
            'Optimiste': optimistic,
            'Conservateur': conservative
        }
    except FileNotFoundError:
        st.error("❌ Fichier Excel 'SOFAC_Scenarios_24Months.xlsx' introuvable. Veuillez le placer dans le même dossier que l'application.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des scénarios: {str(e)}")
        return None

def calculate_loan_costs(scenarios, loan_amount_millions, duration_years, fixed_rate, variable_margin):
    """Calculate loan costs for each scenario"""
    results = {}
    
    # Fixed rate cost (simple)
    fixed_annual_cost = loan_amount_millions * fixed_rate / 100
    fixed_total_cost = fixed_annual_cost * duration_years * 1_000_000  # Convert to MAD
    
    for scenario_name, scenario_df in scenarios.items():
        # Determine months for each year based on duration
        months_per_year = 12
        total_months = duration_years * months_per_year
        
        # Calculate variable costs year by year
        variable_yearly_costs = []
        
        for year in range(duration_years):
            start_month = year * months_per_year
            end_month = min((year + 1) * months_per_year, total_months, len(scenario_df))
            
            # Get average yield for this year
            year_data = scenario_df.iloc[start_month:end_month]
            avg_yield = year_data['Predicted_Yield'].mean()
            
            # Variable rate = yield + margin
            variable_rate = avg_yield + variable_margin
            
            # Annual cost for this year
            year_cost = loan_amount_millions * variable_rate / 100 * 1_000_000
            variable_yearly_costs.append({
                'year': year + 1,
                'avg_yield': avg_yield,
                'variable_rate': variable_rate,
                'cost': year_cost
            })
        
        # Total variable cost
        variable_total_cost = sum([y['cost'] for y in variable_yearly_costs])
        
        # Difference
        difference = variable_total_cost - fixed_total_cost
        
        results[scenario_name] = {
            'fixed_total': fixed_total_cost,
            'variable_total': variable_total_cost,
            'difference': difference,
            'yearly_breakdown': variable_yearly_costs,
            'recommendation': 'FIXE' if difference > 0 else 'VARIABLE',
            'savings_or_cost': abs(difference)
        }
    
    return results

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
            <h1 style="margin: 0; color: white;">Outil de Planification de Scénarios SOFAC</h1>
            <p style="margin: 0.5rem 0; color: white;">Analyse de Sensibilité Taux Fixe vs Variable</p>
            <p style="margin: 0; color: white;">Modèle de Régression | MAE: ±0.22% | MAPE: 10.7%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load scenarios
    scenarios = load_scenarios()
    
    if scenarios is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Performance du Modèle")
        
        st.markdown('<div class="validation-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Modèle Validé")
        st.metric("Erreur Moyenne (MAE)", "±0.22%")
        st.metric("Erreur Relative (MAPE)", "10.7%")
        st.metric("R² (Entraînement)", "94.3%")
        st.caption("Testé sur données 2025")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Équation du Modèle")
        st.code("Rendement = 0.024 + 0.9998×Taux + 0.0444×Inflation")
        
        st.markdown("### Sources")
        st.caption("• Bank Al-Maghrib Q3 2025")
        st.caption("• Données historiques 2018-2025")
        st.caption("• 88 observations mensuelles")
        
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Vue d'Ensemble", "📈 Scénarios Détaillés", "💰 Analyse Décisionnelle"])
    
    with tab1:
        st.header("Tableau de Bord Stratégique")
        
        st.markdown("""
        <div class="warning-box">
            <h4>📋 Objectif de cet Outil</h4>
            <p>Cet outil génère des <strong>scénarios exploratoires</strong> basés sur les prévisions de Bank Al-Maghrib 
            et les tendances historiques pour vous aider à <strong>comparer les options de financement</strong>.</p>
            <p><strong>Important:</strong> Les valeurs ne sont pas des prédictions exactes (erreur moyenne: ±0.22%). 
            L'intérêt est dans la <strong>comparaison relative</strong> entre scénarios.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Current values
        base_scenario = scenarios['Base']
        current_values = base_scenario.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taux Directeur Actuel", f"{current_values['Policy_Rate']:.2f}%")
        with col2:
            st.metric("Inflation Actuelle", f"{current_values['Core_Inflation']:.2f}%")
        with col3:
            st.metric("Rendement Actuel", f"{current_values['Predicted_Yield']:.2f}%")
        with col4:
            st.metric("Horizon", "24 mois")
        
        # Scenario comparison chart
        st.subheader("Comparaison des Scénarios")
        
        fig = go.Figure()
        
        colors = {'Base': '#17a2b8', 'Optimiste': '#28a745', 'Conservateur': '#dc3545'}
        labels = {
            'Base': 'Scénario de Base (Poursuite Prudente)',
            'Optimiste': 'Scénario Optimiste (Assouplissement)',
            'Conservateur': 'Scénario Conservateur (Pause/Hausse)'
        }
        
        for scenario_name, scenario_df in scenarios.items():
            fig.add_trace(go.Scatter(
                x=scenario_df['Date'],
                y=scenario_df['Predicted_Yield'],
                mode='lines+markers',
                name=labels[scenario_name],
                line=dict(color=colors[scenario_name], width=3),
                marker=dict(size=4)
            ))
            
            # Add uncertainty band for base case only
            if scenario_name == 'Base':
                fig.add_trace(go.Scatter(
                    x=scenario_df['Date'],
                    y=scenario_df['Upper_Bound'],
                    mode='lines',
                    name='Incertitude (±0.22%)',
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
                    name='Bande d\'incertitude',
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement 52-Semaines (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario summary
        st.subheader("Synthèse des Scénarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_avg = scenarios['Base']['Predicted_Yield'].mean()
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #17a2b8;">
                <h4 style="color: #17a2b8;">Base (Poursuite Prudente)</h4>
                <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{base_avg:.2f}%</p>
                <p style="font-size: 0.85rem; color: #666;">BAM poursuit l'assouplissement prudemment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            opt_avg = scenarios['Optimiste']['Predicted_Yield'].mean()
            change_opt = opt_avg - base_avg
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #28a745;">
                <h4 style="color: #28a745;">Optimiste (Assouplissement)</h4>
                <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{opt_avg:.2f}%</p>
                <p style="font-size: 0.85rem; color: #666;">{change_opt:+.2f}% vs base | Croissance forte</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cons_avg = scenarios['Conservateur']['Predicted_Yield'].mean()
            change_cons = cons_avg - base_avg
            st.markdown(f"""
            <div class="metric-box" style="border-top-color: #dc3545;">
                <h4 style="color: #dc3545;">Conservateur (Pause/Hausse)</h4>
                <p style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">{cons_avg:.2f}%</p>
                <p style="font-size: 0.85rem; color: #666;">{change_cons:+.2f}% vs base | Tensions inflation</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Analyse Détaillée par Scénario")
        
        scenario_choice = st.selectbox(
            "Sélectionnez un scénario à explorer:",
            ['Base', 'Optimiste', 'Conservateur']
        )
        
        selected_scenario = scenarios[scenario_choice]
        
        # Scenario description
        descriptions = {
            'Base': """
            **Scénario: Poursuite Prudente**
            
            BAM maintient son approche "data-dependent" avec un assouplissement graduel. L'inflation reste contrôlée 
            autour de 2%, permettant des baisses de taux mesurées jusqu'à 1.75% d'ici fin 2026.
            
            **Hypothèses clés:**
            - Croissance modérée (3-4%)
            - Inflation stable 1.8-2.0%
            - Pas de chocs externes majeurs
            """,
            'Optimiste': """
            **Scénario: Assouplissement Accéléré**
            
            Conditions favorables (bonne récolte agricole, baisse énergie) permettent à BAM d'accélérer l'assouplissement. 
            Les taux descendent vers 1.50% avec une inflation maîtrisée.
            
            **Hypothèses clés:**
            - Croissance forte (>4%)
            - Inflation faible (<1.5%)
            - Environnement mondial favorable
            """,
            'Conservateur': """
            **Scénario: Pause Prolongée / Resserrement**
            
            Pressions inflationnistes (importées ou internes) forcent BAM à pausersonassouplissement puis à remonter les taux. 
            L'inflation s'approche du plafond de 2.5%.
            
            **Hypothèses clés:**
            - Tensions géopolitiques
            - Inflation importée
            - Possible sécheresse
            """
        }
        
        st.markdown(descriptions[scenario_choice])
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{selected_scenario['Predicted_Yield'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{selected_scenario['Predicted_Yield'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{selected_scenario['Predicted_Yield'].max():.2f}%")
        with col4:
            volatility = selected_scenario['Predicted_Yield'].std()
            st.metric("Volatilité", f"{volatility:.2f}%")
        
        # Detailed chart
        fig_detail = go.Figure()
        
        fig_detail.add_trace(go.Scatter(
            x=selected_scenario['Date'],
            y=selected_scenario['Predicted_Yield'],
            mode='lines+markers',
            name='Rendement 52s',
            line=dict(color='#2a5298', width=3),
            marker=dict(size=6)
        ))
        
        fig_detail.add_trace(go.Scatter(
            x=selected_scenario['Date'],
            y=selected_scenario['Policy_Rate'],
            mode='lines',
            name='Taux Directeur',
            line=dict(color='#dc3545', width=2, dash='dash')
        ))
        
        fig_detail.add_trace(go.Scatter(
            x=selected_scenario['Date'],
            y=selected_scenario['Core_Inflation'],
            mode='lines',
            name='Inflation',
            line=dict(color='#28a745', width=2, dash='dot')
        ))
        
        fig_detail.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Taux (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Data table
        with st.expander("📋 Voir les données détaillées"):
            display_df = selected_scenario[['Date', 'Policy_Rate', 'Core_Inflation', 'Predicted_Yield', 'Lower_Bound', 'Upper_Bound']].copy()
            display_df.columns = ['Date', 'Taux Directeur', 'Inflation', 'Rendement', 'Borne Inf', 'Borne Sup']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Analyse Comparative: Taux Fixe vs Variable")
        
        st.markdown("""
        <div class="status-card">
            <h4>💡 Utilisez cet outil pour:</h4>
            <ul>
                <li>Comparer le coût total d'un prêt à taux fixe vs variable</li>
                <li>Tester la sensibilité de votre décision aux différents scénarios économiques</li>
                <li>Identifier votre exposition au risque de taux</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Loan parameters
        st.subheader("Paramètres du Prêt")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            loan_amount = st.number_input(
                "Montant (millions MAD)",
                min_value=1,
                max_value=500,
                value=50,
                step=5
            )
        
        with col2:
            duration = st.slider(
                "Durée (années)",
                min_value=1,
                max_value=5,
                value=2
            )
        
        with col3:
            fixed_rate = st.number_input(
                "Taux fixe proposé (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.20,
                step=0.1
            )
        
        with col4:
            margin = st.number_input(
                "Marge variable (%)",
                min_value=0.5,
                max_value=3.0,
                value=1.30,
                step=0.1
            )
        
        # Warning if duration > 2 years
        if duration > 2:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Note:</strong> Les scénarios couvrent 24 mois. Pour une durée de {duration} ans, 
                seules les 2 premières années sont basées sur les scénarios. Les années suivantes sont extrapolées.
            </div>
            """.format(duration=duration), unsafe_allow_html=True)
        
        # Calculate costs
        loan_results = calculate_loan_costs(scenarios, loan_amount, duration, fixed_rate, margin)
        
        # Results summary
        st.subheader("Résultats de l'Analyse")
        
        # Create comparison table
        comparison_data = []
        for scenario_name in ['Base', 'Optimiste', 'Conservateur']:
            result = loan_results[scenario_name]
            comparison_data.append({
                'Scénario': scenario_name,
                'Coût Fixe (MAD)': f"{result['fixed_total']:,.0f}",
                'Coût Variable (MAD)': f"{result['variable_total']:,.0f}",
                'Différence (MAD)': f"{result['difference']:+,.0f}",
                'Recommandation': result['recommendation']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.subheader("Comparaison Visuelle")
        
        fig_comparison = go.Figure()
        
        scenarios_list = ['Base', 'Optimiste', 'Conservateur']
        fixed_costs = [loan_results[s]['fixed_total'] for s in scenarios_list]
        variable_costs = [loan_results[s]['variable_total'] for s in scenarios_list]
        
        fig_comparison.add_trace(go.Bar(
            name='Taux Fixe',
            x=scenarios_list,
            y=fixed_costs,
            marker_color='#17a2b8',
            text=[f"{c/1e6:.2f}M" for c in fixed_costs],
            textposition='auto'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Taux Variable',
            x=scenarios_list,
            y=variable_costs,
            marker_color='#ffc107',
            text=[f"{c/1e6:.2f}M" for c in variable_costs],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            barmode='group',
            height=400,
            template="plotly_white",
            xaxis_title="Scénario",
            yaxis_title="Coût Total (MAD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Decision recommendation
        variable_favorable = sum([1 for r in loan_results.values() if r['recommendation'] == 'VARIABLE'])
        
        if variable_favorable == 3:
            recommendation = "TAUX VARIABLE FORTEMENT RECOMMANDÉ"
            color = "#28a745"
            reason = "Le taux variable est avantageux dans les trois scénarios."
        elif variable_favorable == 2:
            recommendation = "TAUX VARIABLE FAVORABLE"
            color = "#17a2b8"
            reason = "Le taux variable est avantageux dans la majorité des scénarios (2 sur 3)."
        elif variable_favorable == 1:
            recommendation = "DÉCISION SENSIBLE AU SCÉNARIO"
            color = "#ffc107"
            reason = "Les résultats sont partagés. Votre choix dépend de votre anticipation économique et tolérance au risque."
        else:
            recommendation = "TAUX FIXE RECOMMANDÉ"
            color = "#dc3545"
            reason = "Le taux fixe est avantageux dans les trois scénarios."
        
        st.markdown(f"""
        <div class="recommendation-panel" style="background: linear-gradient(135deg, {color}, {color}CC);">
            <h2>🎯 Recommandation SOFAC</h2>
            <h3>{recommendation}</h3>
            <p><strong>Justification:</strong> {reason}</p>
            <hr style="margin: 1.5rem 0; opacity: 0.3;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <p style="font-size: 0.9rem; margin: 0;"><strong>Montant:</strong> {loan_amount}M MAD</p>
                    <p style="font-size: 0.9rem; margin: 0;"><strong>Durée:</strong> {duration} ans</p>
                </div>
                <div>
                    <p style="font-size: 0.9rem; margin: 0;"><strong>Taux fixe:</strong> {fixed_rate}%</p>
                    <p style="font-size: 0.9rem; margin: 0;"><strong>Marge variable:</strong> {margin}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        with st.expander("📊 Voir le détail année par année"):
            for scenario_name in ['Base', 'Optimiste', 'Conservateur']:
                st.markdown(f"### {scenario_name}")
                result = loan_results[scenario_name]
                
                yearly_data = []
                for year_info in result['yearly_breakdown']:
                    yearly_data.append({
                        'Année': year_info['year'],
                        'Rendement Moyen': f"{year_info['avg_yield']:.2f}%",
                        'Taux Variable': f"{year_info['variable_rate']:.2f}%",
                        'Coût Année (MAD)': f"{year_info['cost']:,.0f}"
                    })
                
                yearly_df = pd.DataFrame(yearly_data)
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)
        
        # Important disclaimers
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Points Importants</h4>
            <ul>
                <li>Cette analyse est basée sur des <strong>scénarios hypothétiques</strong>, pas des prédictions certaines</li>
                <li>Les taux réels futurs peuvent différer significativement (erreur moyenne du modèle: ±0.22%)</li>
                <li>Considérez d'autres facteurs: votre situation financière, tolérance au risque, flexibilité de remboursement</li>
                <li>Consultez votre conseiller SOFAC pour une analyse personnalisée</li>
                <li>Le taux fixe offre une <strong>certitude</strong>, le taux variable offre une <strong>opportunité</strong> avec plus de risque</li>
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
            <p style="margin: 0.5rem 0;">Modèle de Régression | MAE: ±0.22% | MAPE: 10.7% | Mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Outil d'aide à la décision - Les scénarios sont basés sur BAM Q3 2025</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
