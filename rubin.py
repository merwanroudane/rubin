import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Rubin Causal Model: Complete Guide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 1.5rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin: 1rem 0;
    }
    .important-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff5f5;
        border-left: 5px solid #ff6b6b;
        padding: 1rem;
        margin: 1rem 0;
    }
    .example-box {
        background-color: #f0fff0;
        border-left: 5px solid #2ca02c;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
sections = [
    "🏠 Introduction",
    "📖 Terminology & Definitions",
    "🎯 Nature of the Model",
    "🧮 Mathematical Framework",
    "⚖️ Assumptions",
    "💼 Applications & Examples",
    "📊 Practical Implementation",
    "🔍 Result Interpretation",
    "📚 Summary & References"
]

selected_section = st.sidebar.selectbox("Choose a section:", sections)

# Main title
st.markdown('<h1 class="main-header">📊 The Rubin Causal Model: A Complete Guide</h1>', unsafe_allow_html=True)

# Section 1: Introduction
if selected_section == "🏠 Introduction":
    st.markdown('<h2 class="section-header">🏠 Introduction to the Rubin Causal Model</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="important-box">
        <h3>What is the Rubin Causal Model?</h3>
        <p>The Rubin Causal Model (RCM), also known as the <strong>Potential Outcomes Framework</strong>, 
        is a statistical framework for causal inference developed by Donald Rubin in the 1970s. 
        It provides a mathematical foundation for understanding causality in observational and experimental data.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Key Features:
        - **Counterfactual thinking**: What would have happened under different treatments?
        - **Individual-level causation**: Focus on individual treatment effects
        - **Mathematical rigor**: Formal framework with clear assumptions
        - **Practical applications**: Widely used in economics, medicine, and social sciences
        """)

        st.markdown("""
        ### 🏛️ Historical Context:
        The RCM emerged from the need to formalize causal inference in statistics. Before Rubin's work, 
        causality was often discussed informally. The RCM provided:
        - A clear definition of causal effects
        - A framework for identifying when causal inference is possible
        - Methods for estimating causal effects from data
        """)

    with col2:
        # Create a conceptual diagram
        fig = go.Figure()

        # Add nodes
        fig.add_trace(go.Scatter(
            x=[0.5, 0.2, 0.8, 0.5],
            y=[0.8, 0.4, 0.4, 0.1],
            mode='markers+text',
            marker=dict(size=[60, 50, 50, 40], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
            text=['Individual<br>Unit', 'Treatment<br>T=1', 'Control<br>T=0', 'Causal<br>Effect'],
            textposition="middle center",
            textfont=dict(color='white', size=10, family='Arial Black'),
            name=""
        ))

        # Add arrows
        fig.add_annotation(x=0.35, y=0.6, ax=0.2, ay=0.4,
                           arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#ff7f0e')
        fig.add_annotation(x=0.65, y=0.6, ax=0.8, ay=0.4,
                           arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#2ca02c')
        fig.add_annotation(x=0.5, y=0.25, ax=0.5, ay=0.4,
                           arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#d62728')

        fig.update_layout(
            title="RCM Conceptual Framework",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

# Section 2: Terminology & Definitions
elif selected_section == "📖 Terminology & Definitions":
    st.markdown('<h2 class="section-header">📖 Terminology & Definitions</h2>', unsafe_allow_html=True)

    # Create tabs for different concepts
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔤 Basic Terms", "🎯 Treatment Concepts", "📊 Outcome Concepts", "⚖️ Effect Measures"])

    with tab1:
        st.markdown('<h3 class="subsection-header">🔤 Fundamental Terminology</h3>', unsafe_allow_html=True)

        terms_data = {
            "Term": ["Unit (i)", "Treatment (T)", "Outcome (Y)", "Assignment Mechanism",
                     "Potential Outcome", "Observed Outcome", "Counterfactual", "Causal Effect"],
            "Definition": [
                "An individual entity (person, firm, country, etc.) that can receive treatment",
                "An intervention or condition that can be applied to a unit (binary: T ∈ {0,1})",
                "The result or response variable we want to measure",
                "The process that determines which units receive which treatments",
                "The outcome that would be observed under a specific treatment",
                "The outcome that is actually observed for a unit",
                "The unobserved potential outcome under the alternative treatment",
                "The difference between potential outcomes under different treatments"
            ],
            "Notation": ["i ∈ {1,2,...,N}", "Tᵢ ∈ {0,1}", "Yᵢ", "P(T|X,Y₀,Y₁)",
                         "Y₀ᵢ, Y₁ᵢ", "Yᵢᵒᵇˢ", "Unobserved", "τᵢ = Y₁ᵢ - Y₀ᵢ"]
        }

        df_terms = pd.DataFrame(terms_data)
        st.dataframe(df_terms, use_container_width=True)

        st.markdown("""
        <div class="important-box">
        <h4>🔑 Key Insight: The Fundamental Problem of Causal Inference</h4>
        <p>We can never observe both Y₁ᵢ and Y₀ᵢ for the same unit at the same time. 
        This is because a unit cannot simultaneously be in both treatment and control states.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<h3 class="subsection-header">🎯 Treatment-Related Concepts</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Treatment Types:
            - **Binary Treatment**: T ∈ {0,1}
              - 0 = Control/No treatment
              - 1 = Treatment/Intervention

            - **Multi-valued Treatment**: T ∈ {0,1,2,...,K}
              - Multiple treatment levels
              - Dose-response relationships

            - **Continuous Treatment**: T ∈ ℝ
              - Treatment intensity varies continuously
              - Examples: drug dosage, spending amounts
            """)

        with col2:
            # Visualization of treatment types
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Binary Treatment", "Multi-valued Treatment", "Continuous Treatment"),
                vertical_spacing=0.1
            )

            # Binary
            fig.add_trace(go.Bar(x=['Control', 'Treatment'], y=[100, 150],
                                 marker_color=['#ff7f0e', '#1f77b4'], name="Binary"),
                          row=1, col=1)

            # Multi-valued
            fig.add_trace(go.Bar(x=['Control', 'Low', 'Medium', 'High'], y=[100, 120, 140, 160],
                                 marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'], name="Multi-valued"),
                          row=2, col=1)

            # Continuous
            x_cont = np.linspace(0, 10, 100)
            y_cont = 100 + 5 * x_cont + np.random.normal(0, 2, 100)
            fig.add_trace(go.Scatter(x=x_cont, y=y_cont, mode='markers',
                                     marker_color='#1f77b4', name="Continuous"),
                          row=3, col=1)

            fig.update_layout(height=600, showlegend=False, title_text="Treatment Types")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<h3 class="subsection-header">📊 Outcome Concepts</h3>', unsafe_allow_html=True)

        st.latex(r'''
        \begin{align}
        Y_i^{obs} &= T_i \cdot Y_{1i} + (1-T_i) \cdot Y_{0i} \\
        &= \begin{cases}
        Y_{1i} & \text{if } T_i = 1 \\
        Y_{0i} & \text{if } T_i = 0
        \end{cases}
        \end{align}
        ''')

        st.markdown("""
        #### 📈 Outcome Relationships:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Potential Outcomes:**
            - Y₀ᵢ: Outcome under control (T=0)
            - Y₁ᵢ: Outcome under treatment (T=1)
            - Both exist but only one is observed

            **Observed Outcome:**
            - What we actually see in the data
            - Switching equation links potential to observed
            - Missing data problem for causal inference
            """)

        with col2:
            # Create potential outcomes visualization
            np.random.seed(42)
            n_units = 50
            y0 = np.random.normal(100, 15, n_units)
            treatment_effect = np.random.normal(10, 5, n_units)
            y1 = y0 + treatment_effect

            fig = go.Figure()

            for i in range(min(10, n_units)):
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[y0[i], y1[i]],
                    mode='lines+markers',
                    line=dict(color='lightblue', width=1),
                    marker=dict(size=6),
                    showlegend=False,
                    opacity=0.7
                ))

            fig.add_trace(go.Scatter(
                x=[0] * n_units, y=y0,
                mode='markers',
                marker=dict(color='#ff7f0e', size=8),
                name='Y₀ (Control)',
                opacity=0.8
            ))

            fig.add_trace(go.Scatter(
                x=[1] * n_units, y=y1,
                mode='markers',
                marker=dict(color='#1f77b4', size=8),
                name='Y₁ (Treatment)',
                opacity=0.8
            ))

            fig.update_layout(
                title="Potential Outcomes for Individual Units",
                xaxis_title="Treatment Status",
                yaxis_title="Outcome",
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Control', 'Treatment'])
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<h3 class="subsection-header">⚖️ Effect Measures</h3>', unsafe_allow_html=True)

        st.markdown("""
        #### Individual Treatment Effect (ITE):
        """)
        st.latex(r'\tau_i = Y_{1i} - Y_{0i}')

        st.markdown("""
        #### Average Treatment Effect (ATE):
        """)
        st.latex(r'''
        \begin{align}
        \text{ATE} &= E[\tau_i] = E[Y_{1i} - Y_{0i}] \\
        &= E[Y_{1i}] - E[Y_{0i}] \\
        &= \mu_1 - \mu_0
        \end{align}
        ''')

        st.markdown("""
        #### Average Treatment Effect on the Treated (ATT):
        """)
        st.latex(r'\text{ATT} = E[Y_{1i} - Y_{0i} | T_i = 1]')

        st.markdown("""
        #### Average Treatment Effect on the Controls (ATC):
        """)
        st.latex(r'\text{ATC} = E[Y_{1i} - Y_{0i} | T_i = 0]')

        # Visualization of different effect measures
        np.random.seed(42)
        n = 1000

        # Generate data with heterogeneous treatment effects
        baseline = np.random.normal(50, 10, n)
        treatment_effect = 5 + 0.3 * baseline + np.random.normal(0, 3, n)

        y0 = baseline
        y1 = baseline + treatment_effect

        # Random assignment
        treatment = np.random.binomial(1, 0.5, n)

        # Calculate effects
        ate = np.mean(treatment_effect)
        att = np.mean(treatment_effect[treatment == 1])
        atc = np.mean(treatment_effect[treatment == 0])

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=treatment_effect,
            nbinsx=30,
            name='Individual Treatment Effects',
            marker_color='lightblue',
            opacity=0.7
        ))

        fig.add_vline(x=ate, line_dash="dash", line_color="red",
                      annotation_text=f"ATE = {ate:.2f}")
        fig.add_vline(x=att, line_dash="dash", line_color="blue",
                      annotation_text=f"ATT = {att:.2f}")
        fig.add_vline(x=atc, line_dash="dash", line_color="green",
                      annotation_text=f"ATC = {atc:.2f}")

        fig.update_layout(
            title="Distribution of Individual Treatment Effects",
            xaxis_title="Treatment Effect",
            yaxis_title="Frequency"
        )

        st.plotly_chart(fig, use_container_width=True)

# Section 3: Nature of the Model
elif selected_section == "🎯 Nature of the Model":
    st.markdown('<h2 class="section-header">🎯 Nature of the Rubin Causal Model</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 When to Use", "🌍 Where to Apply", "❓ Why Use RCM"])

    with tab1:
        st.markdown('<h3 class="subsection-header">🎯 When to Use the Rubin Causal Model</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="important-box">
            <h4>🎯 Primary Use Cases:</h4>
            <ul>
                <li><strong>Policy Evaluation</strong>: Assessing the impact of government programs</li>
                <li><strong>Medical Research</strong>: Evaluating treatment effectiveness</li>
                <li><strong>Marketing</strong>: Measuring advertising campaign effects</li>
                <li><strong>Education</strong>: Analyzing educational interventions</li>
                <li><strong>Labor Economics</strong>: Studying job training programs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            #### 🔍 Research Questions Suitable for RCM:

            1. **Causal Questions**: "What is the effect of X on Y?"
            2. **Counterfactual Questions**: "What would have happened if...?"
            3. **Policy Questions**: "Should we implement this intervention?"
            4. **Comparative Questions**: "Which treatment works better?"
            """)

            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ When NOT to Use RCM:</h4>
            <ul>
                <li><strong>Prediction Tasks</strong>: When you only care about forecasting</li>
                <li><strong>Descriptive Analysis</strong>: When you just want to describe patterns</li>
                <li><strong>Correlation Studies</strong>: When causation is not the goal</li>
                <li><strong>Impossible Interventions</strong>: When treatment cannot be manipulated</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Decision tree for when to use RCM
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=["Research Goal", "Causal Inference", "Prediction", "Use RCM", "Use ML/Stats"],
                    color=["#1f77b4", "#2ca02c", "#ff7f0e", "#2ca02c", "#ff7f0e"]
                ),
                link=dict(
                    source=[0, 0, 1, 2],
                    target=[1, 2, 3, 4],
                    value=[70, 30, 70, 30],
                    color=["rgba(44, 160, 44, 0.3)", "rgba(255, 127, 14, 0.3)",
                           "rgba(44, 160, 44, 0.5)", "rgba(255, 127, 14, 0.5)"]
                )
            ))

            fig.update_layout(title_text="When to Use RCM: Decision Flow", font_size=10, height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<h3 class="subsection-header">🌍 Where to Apply the RCM</h3>', unsafe_allow_html=True)

        # Create a comprehensive field overview
        fields_data = {
            "Field": ["Economics", "Medicine", "Education", "Marketing", "Public Policy", "Psychology"],
            "Applications": [
                "Labor market interventions, fiscal policy, trade effects",
                "Clinical trials, treatment effectiveness, drug evaluation",
                "Educational programs, teaching methods, school policies",
                "Advertising campaigns, pricing strategies, product launches",
                "Social programs, regulation effects, welfare policies",
                "Behavioral interventions, therapy effectiveness, cognitive training"
            ],
            "Example Studies": [
                "Job training programs, minimum wage effects",
                "Randomized controlled trials, observational studies",
                "Class size reduction, technology in classrooms",
                "A/B testing, targeted advertising",
                "Conditional cash transfers, housing vouchers",
                "Cognitive behavioral therapy, mindfulness interventions"
            ]
        }

        df_fields = pd.DataFrame(fields_data)

        for i, row in df_fields.iterrows():
            with st.expander(f"📍 {row['Field']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Applications:**\n{row['Applications']}")
                with col2:
                    st.markdown(f"**Example Studies:**\n{row['Example Studies']}")

        # Geographic visualization of RCM usage
        st.markdown("#### 🗺️ Global Usage of RCM by Region")

        regions_data = {
            'Region': ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Africa', 'Middle East'],
            'Usage_Score': [95, 85, 70, 60, 45, 50],
            'Primary_Fields': ['Economics, Medicine', 'Public Policy, Medicine', 'Economics, Education',
                               'Public Policy, Economics', 'Development, Health', 'Policy, Development']
        }

        fig = px.bar(
            regions_data,
            x='Region',
            y='Usage_Score',
            color='Usage_Score',
            color_continuous_scale='viridis',
            title="RCM Adoption by Region (Usage Score 0-100)"
        )

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<h3 class="subsection-header">❓ Why Use the Rubin Causal Model</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="important-box">
            <h4>🎯 Advantages of RCM:</h4>
            <ol>
                <li><strong>Conceptual Clarity</strong>: Clear definition of causal effects</li>
                <li><strong>Mathematical Rigor</strong>: Formal statistical framework</li>
                <li><strong>Individual Focus</strong>: Accounts for individual heterogeneity</li>
                <li><strong>Assumption Transparency</strong>: Makes assumptions explicit</li>
                <li><strong>Policy Relevance</strong>: Direct connection to decision-making</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            #### 🔬 Scientific Benefits:
            - **Reproducibility**: Clear methodology for replication
            - **Falsifiability**: Testable assumptions
            - **Generalizability**: Framework applies across disciplines
            - **Comparability**: Standard approach enables meta-analysis
            """)

        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Limitations of RCM:</h4>
            <ol>
                <li><strong>Unobservability</strong>: Counterfactuals are never observed</li>
                <li><strong>Strong Assumptions</strong>: SUTVA, ignorability requirements</li>
                <li><strong>Binary Focus</strong>: Originally designed for binary treatments</li>
                <li><strong>Static Framework</strong>: Doesn't handle dynamic treatments well</li>
                <li><strong>External Validity</strong>: Results may not generalize</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            #### 🔄 Alternative Approaches:
            - **Structural Causal Models**: Pearl's causal diagrams
            - **Instrumental Variables**: For endogeneity issues
            - **Regression Discontinuity**: For quasi-experimental settings
            - **Difference-in-Differences**: For panel data settings
            """)

        # Comparison chart
        comparison_data = {
            'Aspect': ['Conceptual Clarity', 'Mathematical Rigor', 'Practical Implementation',
                       'Assumption Transparency', 'Computational Complexity'],
            'RCM': [9, 9, 7, 9, 6],
            'Traditional Regression': [5, 6, 9, 4, 8],
            'Machine Learning': [4, 7, 8, 3, 5]
        }

        fig = go.Figure()

        for method in ['RCM', 'Traditional Regression', 'Machine Learning']:
            fig.add_trace(go.Scatterpolar(
                r=comparison_data[method],
                theta=comparison_data['Aspect'],
                fill='toself',
                name=method,
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Method Comparison (1-10 scale)"
        )

        st.plotly_chart(fig, use_container_width=True)

# Section 4: Mathematical Framework
elif selected_section == "🧮 Mathematical Framework":
    st.markdown('<h2 class="section-header">🧮 Mathematical Framework</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔢 Basic Setup", "📊 Identification", "📈 Estimation", "🎯 Advanced Topics"])

    with tab1:
        st.markdown('<h3 class="subsection-header">🔢 Basic Mathematical Setup</h3>', unsafe_allow_html=True)

        st.markdown("""
        #### 1. Population and Units
        """)
        st.latex(r'''
        \text{Population: } \mathcal{U} = \{1, 2, \ldots, N\}
        ''')

        st.markdown("""
        #### 2. Treatment Assignment
        """)
        st.latex(r'''
        T_i \in \{0, 1\} \quad \forall i \in \mathcal{U}
        ''')

        st.markdown("""
        #### 3. Potential Outcomes
        """)
        st.latex(r'''
        \begin{align}
        Y_{0i} &: \text{Potential outcome under control} \\
        Y_{1i} &: \text{Potential outcome under treatment} \\
        \mathbf{Y}_i &= (Y_{0i}, Y_{1i}) \text{ (potential outcome vector)}
        \end{align}
        ''')

        st.markdown("""
        #### 4. Observed Outcome (Switching Equation)
        """)
        st.latex(r'''
        Y_i^{obs} = T_i Y_{1i} + (1-T_i) Y_{0i}
        ''')

        st.markdown("""
        #### 5. Individual Treatment Effect
        """)
        st.latex(r'''
        \tau_i = Y_{1i} - Y_{0i}
        ''')

        # Interactive visualization of the switching equation
        st.markdown("#### 📊 Interactive Switching Equation Visualization")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Adjust Parameters:**")
            n_units = st.slider("Number of units", 10, 100, 50)
            y0_mean = st.slider("Y₀ mean", 80, 120, 100)
            treatment_effect = st.slider("Average treatment effect", 0, 20, 10)
            noise_level = st.slider("Noise level", 1, 10, 5)

        with col2:
            np.random.seed(42)
            y0 = np.random.normal(y0_mean, noise_level, n_units)
            y1 = y0 + np.random.normal(treatment_effect, 3, n_units)
            treatment = np.random.binomial(1, 0.5, n_units)
            y_obs = treatment * y1 + (1 - treatment) * y0

            fig = go.Figure()

            # Potential outcomes
            fig.add_trace(go.Scatter(
                x=list(range(n_units)), y=y0,
                mode='markers', name='Y₀ (Control potential)',
                marker=dict(color='lightcoral', size=6, opacity=0.7)
            ))

            fig.add_trace(go.Scatter(
                x=list(range(n_units)), y=y1,
                mode='markers', name='Y₁ (Treatment potential)',
                marker=dict(color='lightblue', size=6, opacity=0.7)
            ))

            # Observed outcomes
            colors = ['red' if t == 0 else 'blue' for t in treatment]
            fig.add_trace(go.Scatter(
                x=list(range(n_units)), y=y_obs,
                mode='markers', name='Y^obs (Observed)',
                marker=dict(color=colors, size=10, symbol='diamond')
            ))

            fig.update_layout(
                title="Switching Equation: Observed vs Potential Outcomes",
                xaxis_title="Unit ID",
                yaxis_title="Outcome Value",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<h3 class="subsection-header">📊 Identification of Causal Effects</h3>', unsafe_allow_html=True)

        st.markdown("""
        #### The Fundamental Problem of Causal Inference
        """)

        st.latex(r'''
        \text{We observe: } \{(Y_i^{obs}, T_i)\}_{i=1}^N \\
        \text{We want: } E[Y_{1i} - Y_{0i}]
        ''')

        st.markdown("""
        <div class="warning-box">
        <h4>🚫 The Problem:</h4>
        <p>For each unit i, we observe either Y₁ᵢ or Y₀ᵢ, but never both!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### Naive Comparison (Biased!)
        """)
        st.latex(r'''
        \begin{align}
        E[Y_i^{obs} | T_i = 1] - E[Y_i^{obs} | T_i = 0] &= E[Y_{1i} | T_i = 1] - E[Y_{0i} | T_i = 0] \\
        &= \underbrace{E[Y_{1i} | T_i = 1] - E[Y_{0i} | T_i = 1]}_{\text{ATT}} \\
        &\quad + \underbrace{E[Y_{0i} | T_i = 1] - E[Y_{0i} | T_i = 0]}_{\text{Selection Bias}}
        \end{align}
        ''')

        st.markdown("""
        #### Selection Bias Decomposition
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.latex(r'''
            \begin{align}
            \text{Observed Difference} &= \text{True Effect} + \text{Bias} \\
            E[Y|T=1] - E[Y|T=0] &= \text{ATT} + \text{Selection Bias}
            \end{align}
            ''')

            st.markdown("""
            **Selection Bias Sources:**
            - Self-selection into treatment
            - Systematic assignment by administrators
            - Confounding variables
            - Unobserved heterogeneity
            """)

        with col2:
            # Visualization of selection bias
            np.random.seed(42)
            n = 1000

            # Create selection bias scenario
            ability = np.random.normal(0, 1, n)
            prob_treatment = 1 / (1 + np.exp(-ability))  # Higher ability → higher treatment probability
            treatment = np.random.binomial(1, prob_treatment, n)

            y0 = 50 + 10 * ability + np.random.normal(0, 5, n)
            y1 = y0 + 15 + np.random.normal(0, 3, n)  # True treatment effect = 15

            y_obs = treatment * y1 + (1 - treatment) * y0

            # Calculate means
            y_treated = np.mean(y_obs[treatment == 1])
            y_control = np.mean(y_obs[treatment == 0])
            naive_diff = y_treated - y_control

            # True ATT
            att_true = 15  # We know this from construction

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=y_obs[treatment == 0], name='Control Group',
                opacity=0.7, marker_color='red', nbinsx=30
            ))

            fig.add_trace(go.Histogram(
                x=y_obs[treatment == 1], name='Treatment Group',
                opacity=0.7, marker_color='blue', nbinsx=30
            ))

            fig.add_vline(x=y_control, line_dash="dash", line_color="red",
                          annotation_text=f"Control Mean: {y_control:.1f}")
            fig.add_vline(x=y_treated, line_dash="dash", line_color="blue",
                          annotation_text=f"Treatment Mean: {y_treated:.1f}")

            fig.update_layout(
                title=f"Selection Bias Example<br>Naive Difference: {naive_diff:.1f} | True ATT: {att_true}",
                xaxis_title="Outcome",
                yaxis_title="Frequency",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Identification Strategies
        """)

        strategies = {
            "Strategy": ["Randomized Experiment", "Unconfoundedness", "Instrumental Variables",
                         "Regression Discontinuity", "Difference-in-Differences"],
            "Key Assumption": [
                "Random assignment: T ⊥ (Y₀, Y₁)",
                "No unmeasured confounders: T ⊥ (Y₀, Y₁) | X",
                "Valid instrument: Z affects Y only through T",
                "Continuity at cutoff",
                "Parallel trends"
            ],
            "Advantage": [
                "Gold standard, unbiased",
                "Works with observational data",
                "Handles endogeneity",
                "Natural experiment",
                "Controls for time-invariant confounders"
            ]
        }

        df_strategies = pd.DataFrame(strategies)
        st.dataframe(df_strategies, use_container_width=True)

    with tab3:
        st.markdown('<h3 class="subsection-header">📈 Estimation Methods</h3>', unsafe_allow_html=True)

        estimation_method = st.selectbox(
            "Choose estimation method to explore:",
            ["Simple Difference in Means", "Regression Adjustment", "Propensity Score Methods", "Matching"]
        )

        if estimation_method == "Simple Difference in Means":
            st.markdown("""
            #### Simple Difference in Means (Under Randomization)
            """)

            st.latex(r'''
            \hat{\tau}_{ATE} = \frac{1}{N_1} \sum_{i: T_i = 1} Y_i - \frac{1}{N_0} \sum_{i: T_i = 0} Y_i
            ''')

            st.markdown("""
            **Standard Error:**
            """)
            st.latex(r'''
            SE(\hat{\tau}_{ATE}) = \sqrt{\frac{S_1^2}{N_1} + \frac{S_0^2}{N_0}}
            ''')

            # Simulation
            st.markdown("#### 🎲 Simulation: Randomized Experiment")

            col1, col2 = st.columns([1, 2])

            with col1:
                n_sim = st.slider("Sample size", 50, 500, 200)
                true_effect = st.slider("True treatment effect", 0, 20, 10)
                p_treat = st.slider("Treatment probability", 0.1, 0.9, 0.5)

            with col2:
                np.random.seed(42)
                y0_sim = np.random.normal(100, 15, n_sim)
                y1_sim = y0_sim + true_effect + np.random.normal(0, 5, n_sim)
                t_sim = np.random.binomial(1, p_treat, n_sim)
                y_obs_sim = t_sim * y1_sim + (1 - t_sim) * y0_sim

                # Estimate
                y_treat_mean = np.mean(y_obs_sim[t_sim == 1])
                y_control_mean = np.mean(y_obs_sim[t_sim == 0])
                ate_estimate = y_treat_mean - y_control_mean

                # Standard error
                s1_sq = np.var(y_obs_sim[t_sim == 1], ddof=1)
                s0_sq = np.var(y_obs_sim[t_sim == 0], ddof=1)
                n1 = np.sum(t_sim)
                n0 = np.sum(1 - t_sim)
                se_ate = np.sqrt(s1_sq / n1 + s0_sq / n0)

                # Confidence interval
                ci_lower = ate_estimate - 1.96 * se_ate
                ci_upper = ate_estimate + 1.96 * se_ate

                st.markdown(f"""
                **Results:**
                - True ATE: {true_effect}
                - Estimated ATE: {ate_estimate:.2f}
                - Standard Error: {se_ate:.2f}
                - 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]
                - Coverage: {'✅' if ci_lower <= true_effect <= ci_upper else '❌'}
                """)

        elif estimation_method == "Regression Adjustment":
            st.markdown("""
            #### Regression Adjustment
            """)

            st.latex(r'''
            Y_i = \alpha + \tau T_i + \beta X_i + \epsilon_i
            ''')

            st.markdown("""
            **Under unconfoundedness:** $(Y_0, Y_1) ⊥ T | X$
            """)

            st.latex(r'''
            \hat{\tau}_{ATE} = \frac{1}{N} \sum_{i=1}^N [\hat{m}(1, X_i) - \hat{m}(0, X_i)]
            ''')

            st.markdown("""
            where $\hat{m}(t, x)$ is the estimated conditional mean function.
            """)

            # Interactive regression adjustment example
            st.markdown("#### 📊 Interactive Regression Adjustment")

            # Generate data with confounding
            np.random.seed(42)
            n = 500
            x = np.random.normal(0, 1, n)
            prob_t = 1 / (1 + np.exp(-0.5 * x))  # X affects treatment
            t = np.random.binomial(1, prob_t, n)
            y = 50 + 10 * t + 5 * x + np.random.normal(0, 3, n)  # True effect = 10

            # Naive estimate
            naive_est = np.mean(y[t == 1]) - np.mean(y[t == 0])

            # Regression estimate
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression()
            X_reg = np.column_stack([t, x])
            reg.fit(X_reg, y)
            reg_est = reg.coef_[0]

            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(x=x, y=y, color=t.astype(str),
                                 title="Confounded Data",
                                 labels={'x': 'Covariate X', 'y': 'Outcome Y', 'color': 'Treatment'})
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"""
                **Comparison of Estimates:**
                - True Effect: 10.0
                - Naive Estimate: {naive_est:.2f}
                - Regression Estimate: {reg_est:.2f}

                **Bias:**
                - Naive Bias: {naive_est - 10:.2f}
                - Regression Bias: {reg_est - 10:.2f}
                """)

        elif estimation_method == "Propensity Score Methods":
            st.markdown("""
            #### Propensity Score Methods
            """)

            st.markdown("""
            **Propensity Score:** $e(X) = P(T = 1 | X)$
            """)

            st.latex(r'''
            \text{If } (Y_0, Y_1) \perp T | X, \text{ then } (Y_0, Y_1) \perp T | e(X)
            ''')

            st.markdown("""
            **Estimation approaches:**
            1. **Matching**: Match treated and control units with similar propensity scores
            2. **Stratification**: Stratify by propensity score quintiles
            3. **IPTW**: Inverse probability of treatment weighting
            4. **Doubly Robust**: Combine with outcome regression
            """)

            # Propensity score visualization
            np.random.seed(42)
            n = 500
            x1 = np.random.normal(0, 1, n)
            x2 = np.random.normal(0, 1, n)

            # True propensity score
            logit_ps = -0.5 + 0.8 * x1 + 0.6 * x2
            ps_true = 1 / (1 + np.exp(-logit_ps))
            t = np.random.binomial(1, ps_true, n)

            # Outcome with treatment effect = 8
            y = 50 + 8 * t + 3 * x1 + 2 * x2 + np.random.normal(0, 2, n)

            # Estimate propensity scores
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression()
            X_ps = np.column_stack([x1, x2])
            lr.fit(X_ps, t)
            ps_est = lr.predict_proba(X_ps)[:, 1]

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("True Propensity Scores", "Estimated Propensity Scores"))

            # True PS
            fig.add_trace(go.Histogram(x=ps_true[t == 0], name='Control (True)',
                                       marker_color='red', opacity=0.7, nbinsx=20), row=1, col=1)
            fig.add_trace(go.Histogram(x=ps_true[t == 1], name='Treated (True)',
                                       marker_color='blue', opacity=0.7, nbinsx=20), row=1, col=1)

            # Estimated PS
            fig.add_trace(go.Histogram(x=ps_est[t == 0], name='Control (Est)',
                                       marker_color='red', opacity=0.7, nbinsx=20, showlegend=False), row=1, col=2)
            fig.add_trace(go.Histogram(x=ps_est[t == 1], name='Treated (Est)',
                                       marker_color='blue', opacity=0.7, nbinsx=20, showlegend=False), row=1, col=2)

            fig.update_layout(title="Propensity Score Distribution", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)

        elif estimation_method == "Matching":
            st.markdown("""
            #### Matching Methods
            """)

            st.markdown("""
            **Basic Idea:** Find control units that are "similar" to treated units

            **Distance Metrics:**
            - Euclidean distance on covariates
            - Mahalanobis distance
            - Propensity score distance
            """)

            st.latex(r'''
            \hat{\tau}_{ATT} = \frac{1}{N_1} \sum_{i: T_i = 1} \left[ Y_i - \frac{1}{|J(i)|} \sum_{j \in J(i)} Y_j \right]
            ''')

            st.markdown("where $J(i)$ is the set of matches for unit $i$.")

            # Matching visualization
            st.markdown("#### 🎯 Matching Visualization")

            np.random.seed(42)
            n_treat = 20
            n_control = 50

            # Generate treated units
            x_treat = np.random.normal(2, 1, n_treat)
            y_treat = 60 + 10 + 5 * x_treat + np.random.normal(0, 2, n_treat)  # +10 treatment effect

            # Generate control units
            x_control = np.random.normal(0, 1.5, n_control)
            y_control = 60 + 5 * x_control + np.random.normal(0, 2, n_control)

            # Simple nearest neighbor matching
            matches = []
            for i, x_t in enumerate(x_treat):
                distances = np.abs(x_control - x_t)
                match_idx = np.argmin(distances)
                matches.append((i, match_idx, distances[match_idx]))

            fig = go.Figure()

            # Control units
            fig.add_trace(go.Scatter(
                x=x_control, y=y_control,
                mode='markers', name='Control Pool',
                marker=dict(color='lightcoral', size=8, opacity=0.6)
            ))

            # Treated units
            fig.add_trace(go.Scatter(
                x=x_treat, y=y_treat,
                mode='markers', name='Treated Units',
                marker=dict(color='blue', size=10)
            ))

            # Matched control units
            matched_indices = [m[1] for m in matches]
            fig.add_trace(go.Scatter(
                x=x_control[matched_indices], y=y_control[matched_indices],
                mode='markers', name='Matched Controls',
                marker=dict(color='red', size=10, symbol='diamond')
            ))

            # Draw matching lines
            for i, (t_idx, c_idx, dist) in enumerate(matches[:10]):  # Show first 10 matches
                fig.add_trace(go.Scatter(
                    x=[x_treat[t_idx], x_control[c_idx]],
                    y=[y_treat[t_idx], y_control[c_idx]],
                    mode='lines', line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False, opacity=0.5
                ))

            fig.update_layout(
                title="Nearest Neighbor Matching on Covariate X",
                xaxis_title="Covariate X",
                yaxis_title="Outcome Y"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate matching estimate
            att_estimate = np.mean(y_treat) - np.mean(y_control[matched_indices])
            st.markdown(f"""
            **Matching Results:**
            - ATT Estimate: {att_estimate:.2f}
            - True ATT: 10.0
            - Bias: {att_estimate - 10:.2f}
            """)

    with tab4:
        st.markdown('<h3 class="subsection-header">🎯 Advanced Topics</h3>', unsafe_allow_html=True)

        advanced_topic = st.selectbox(
            "Choose advanced topic:",
            ["Heterogeneous Treatment Effects", "Multiple Treatments", "Mediation Analysis", "Sensitivity Analysis"]
        )

        if advanced_topic == "Heterogeneous Treatment Effects":
            st.markdown("""
            #### Heterogeneous Treatment Effects (HTE)
            """)

            st.markdown("""
            **Individual treatment effects vary:** $τᵢ = Y₁ᵢ - Y₀ᵢ ≠ τⱼ$ for $i ≠ j$
            """)

            st.latex(r'''
            \begin{align}
            \text{CATE}(x) &= E[Y_{1i} - Y_{0i} | X_i = x] \\
            &= E[Y_{1i} | X_i = x] - E[Y_{0i} | X_i = x]
            \end{align}
            ''')

            # HTE visualization
            np.random.seed(42)
            n = 500
            x = np.random.uniform(-2, 2, n)

            # Heterogeneous treatment effects
            tau_x = 5 + 3 * x + 0.5 * x ** 2  # Treatment effect varies with X

            t = np.random.binomial(1, 0.5, n)
            y0 = 50 + 2 * x + np.random.normal(0, 3, n)
            y1 = y0 + tau_x + np.random.normal(0, 2, n)
            y_obs = t * y1 + (1 - t) * y0

            # Estimate CATE using local linear regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_predict

            # Separate models for treated and control
            rf_1 = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_0 = RandomForestRegressor(n_estimators=100, random_state=42)

            # Fit models
            rf_1.fit(x[t == 1].reshape(-1, 1), y_obs[t == 1])
            rf_0.fit(x[t == 0].reshape(-1, 1), y_obs[t == 0])

            # Predict CATE
            x_grid = np.linspace(-2, 2, 100)
            mu1_pred = rf_1.predict(x_grid.reshape(-1, 1))
            mu0_pred = rf_0.predict(x_grid.reshape(-1, 1))
            cate_pred = mu1_pred - mu0_pred

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Data by Treatment", "Estimated CATE"))

            # Data plot
            fig.add_trace(go.Scatter(
                x=x[t == 0], y=y_obs[t == 0],
                mode='markers', name='Control',
                marker=dict(color='red', opacity=0.6)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=x[t == 1], y=y_obs[t == 1],
                mode='markers', name='Treated',
                marker=dict(color='blue', opacity=0.6)
            ), row=1, col=1)

            # CATE plot
            fig.add_trace(go.Scatter(
                x=x_grid, y=tau_x[:100],  # True CATE
                mode='lines', name='True CATE',
                line=dict(color='black', width=3)
            ), row=1, col=2)

            fig.add_trace(go.Scatter(
                x=x_grid, y=cate_pred,
                mode='lines', name='Estimated CATE',
                line=dict(color='purple', width=2, dash='dash')
            ), row=1, col=2)

            fig.update_layout(title="Heterogeneous Treatment Effects")
            st.plotly_chart(fig, use_container_width=True)

        elif advanced_topic == "Multiple Treatments":
            st.markdown("""
            #### Multiple Treatments
            """)

            st.markdown("""
            **Setup:** $T_i ∈ \{0, 1, 2, ..., K\}$

            **Potential Outcomes:** $(Y_{0i}, Y_{1i}, Y_{2i}, ..., Y_{Ki})$
            """)

            st.latex(r'''
            \begin{align}
            \tau_{k,j} &= E[Y_{ki} - Y_{ji}] \quad \text{(Pairwise comparison)} \\
            \tau_k &= E[Y_{ki} - Y_{0i}] \quad \text{(Comparison to control)}
            \end{align}
            ''')

            # Multiple treatment visualization
            np.random.seed(42)
            n = 300
            treatments = np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.25, 0.25, 0.2])

            # Different treatment effects
            effects = {0: 0, 1: 8, 2: 12, 3: 15}

            y_outcomes = []
            for t in treatments:
                y = 50 + effects[t] + np.random.normal(0, 5)
                y_outcomes.append(y)

            y_outcomes = np.array(y_outcomes)

            fig = go.Figure()

            colors = ['red', 'blue', 'green', 'orange']
            for k in range(4):
                mask = treatments == k
                fig.add_trace(go.Box(
                    y=y_outcomes[mask],
                    name=f'Treatment {k}',
                    marker_color=colors[k]
                ))

            fig.update_layout(
                title="Multiple Treatment Comparison",
                yaxis_title="Outcome",
                xaxis_title="Treatment Group"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Pairwise comparisons
            st.markdown("#### Pairwise Treatment Effects")

            pairwise_effects = []
            for k in range(1, 4):
                for j in range(k):
                    effect_kj = np.mean(y_outcomes[treatments == k]) - np.mean(y_outcomes[treatments == j])
                    pairwise_effects.append({
                        'Comparison': f'T{k} vs T{j}',
                        'Effect': effect_kj,
                        'True Effect': effects[k] - effects[j]
                    })

            df_pairwise = pd.DataFrame(pairwise_effects)
            st.dataframe(df_pairwise, use_container_width=True)

        elif advanced_topic == "Mediation Analysis":
            st.markdown("""
            #### Mediation Analysis
            """)

            st.markdown("""
            **Research Question:** How does treatment affect outcome? Through what mechanisms?

            **Components:**
            - **Direct Effect**: T → Y (not through mediator)
            - **Indirect Effect**: T → M → Y (through mediator)
            - **Total Effect**: Direct + Indirect
            """)

            st.latex(r'''
            \begin{align}
            \text{Total Effect} &= E[Y_i(1, M_i(1)) - Y_i(0, M_i(0))] \\
            \text{Direct Effect} &= E[Y_i(1, m) - Y_i(0, m)] \\
            \text{Indirect Effect} &= E[Y_i(t, M_i(1)) - Y_i(t, M_i(0))]
            \end{align}
            ''')

            # Mediation path diagram
            fig = go.Figure()

            # Add nodes
            nodes = {
                'T': (0.1, 0.5),
                'M': (0.5, 0.7),
                'Y': (0.9, 0.5),
                'U': (0.5, 0.3)
            }

            for node, (x, y) in nodes.items():
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=40, color='lightblue'),
                    text=node,
                    textposition="middle center",
                    showlegend=False
                ))

            # Add arrows
            arrows = [
                ('T', 'M', 'α'),
                ('T', 'Y', 'τ (direct)'),
                ('M', 'Y', 'β'),
                ('U', 'M', ''),
                ('U', 'Y', '')
            ]

            for start, end, label in arrows:
                x0, y0 = nodes[start]
                x1, y1 = nodes[end]

                fig.add_annotation(
                    x=x1, y=y1, ax=x0, ay=y0,
                    arrowhead=2, arrowsize=1, arrowwidth=2,
                    text=label, textangle=0
                )

            fig.update_layout(
                title="Mediation Path Diagram",
                xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        elif advanced_topic == "Sensitivity Analysis":
            st.markdown("""
            #### Sensitivity Analysis
            """)

            st.markdown("""
            **Question:** How sensitive are our results to violations of key assumptions?

            **Rosenbaum's Γ (Gamma):** Measure of hidden bias
            """)

            st.latex(r'''
            \frac{1}{\Gamma} \leq \frac{\pi_{ij}}{1 - \pi_{ij}} \cdot \frac{1 - \pi_{kl}}{\pi_{kl}} \leq \Gamma
            ''')

            st.markdown("""
            where $π_{ij}$ is the probability that unit $i$ receives treatment rather than unit $j$.

            **Interpretation:**
            - Γ = 1: No hidden bias (randomized experiment)
            - Γ > 1: Units may differ by factor of Γ in odds of treatment
            """)

            # Sensitivity analysis simulation
            gammas = np.linspace(1, 3, 20)
            p_values = []

            np.random.seed(42)
            for gamma in gammas:
                # Simulate bias
                n = 200
                u = np.random.normal(0, 1, n)  # Unobserved confounder

                # Treatment probability depends on unobserved confounder
                logit_p = 0.5 * u
                prob_t = 1 / (1 + np.exp(-logit_p))

                # Add bias proportional to gamma
                prob_t_biased = prob_t * gamma / (1 + prob_t * (gamma - 1))
                t = np.random.binomial(1, prob_t_biased, n)

                # Outcome depends on both treatment and confounder
                y = 50 + 5 * t + 3 * u + np.random.normal(0, 2, n)

                # Test statistic (simplified)
                if np.sum(t) > 0 and np.sum(1 - t) > 0:
                    t_stat, p_val = stats.ttest_ind(y[t == 1], y[t == 0])
                    p_values.append(p_val)
                else:
                    p_values.append(1.0)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=gammas, y=p_values,
                mode='lines+markers',
                name='p-value',
                line=dict(color='blue', width=2)
            ))

            fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                          annotation_text="α = 0.05")

            fig.update_layout(
                title="Sensitivity Analysis: p-value vs. Hidden Bias (Γ)",
                xaxis_title="Gamma (Hidden Bias)",
                yaxis_title="p-value"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - Find the critical Γ where p-value crosses significance threshold
            - This tells us how large hidden bias needs to be to explain away the effect
            - Larger critical Γ = more robust results
            """)

# Section 5: Assumptions
elif selected_section == "⚖️ Assumptions":
    st.markdown('<h2 class="section-header">⚖️ Assumptions of the Rubin Causal Model</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 SUTVA", "🎲 Ignorability", "🔄 Overlap", "📊 Testing Assumptions"])

    with tab1:
        st.markdown('<h3 class="subsection-header">🎯 SUTVA: Stable Unit Treatment Value Assumption</h3>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="important-box">
        <h4>📋 SUTVA Definition</h4>
        <p><strong>SUTVA</strong> consists of two components:</p>
        <ol>
            <li><strong>No Interference</strong>: Unit i's outcome depends only on unit i's treatment</li>
            <li><strong>No Hidden Variations</strong>: For each unit, there is only one version of each treatment</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 1. No Interference (Non-Interference)
            """)
            st.latex(r'''
            Y_i(t_1, t_2, \ldots, t_N) = Y_i(t_i) \quad \forall i
            ''')

            st.markdown("""
            **Violations:**
            - **Spillover effects**: Vaccination reduces disease for non-vaccinated
            - **General equilibrium**: Job training affects wages of non-participants  
            - **Social interactions**: Peer effects in education
            - **Market interactions**: Advertising affects competitors
            """)

            st.markdown("""
            #### Examples of Interference:
            """)

            interference_examples = {
                "Context": ["Vaccination", "Education", "Job Training", "Social Media"],
                "Direct Effect": ["Reduces own infection risk", "Improves own test scores",
                                  "Increases own wages", "Changes own behavior"],
                "Spillover Effect": ["Reduces others' infection risk", "Peer learning effects",
                                     "Affects labor market wages", "Influences friends' behavior"],
                "SUTVA Violation?": ["Yes", "Yes", "Yes", "Yes"]
            }

            df_interference = pd.DataFrame(interference_examples)
            st.dataframe(df_interference, use_container_width=True)

        with col2:
            st.markdown("""
            #### 2. No Hidden Variations
            """)
            st.latex(r'''
            \text{If } T_i = t, \text{ then } Y_i^{obs} = Y_i(t)
            ''')

            st.markdown("""
            **What this means:**
            - Treatment "1" means the same thing for all units
            - No different versions, intensities, or implementations
            - Treatment is well-defined and consistent
            """)

            st.markdown("""
            **Violations:**
            - **Different dosages**: "Treatment" could be 10mg or 50mg
            - **Implementation variation**: Different teachers, different quality
            - **Timing differences**: Morning vs evening treatment
            - **Context variation**: Urban vs rural implementation
            """)

            # Visualization of SUTVA violations
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("No Interference", "Interference Present"),
                vertical_spacing=0.3
            )

            # No interference scenario
            np.random.seed(42)
            n_units = 10
            x_pos = np.arange(n_units)
            treatment = np.random.binomial(1, 0.5, n_units)

            # Outcomes without interference
            y_no_interference = 50 + 10 * treatment + np.random.normal(0, 2, n_units)

            colors_no_int = ['red' if t == 0 else 'blue' for t in treatment]
            fig.add_trace(go.Bar(
                x=x_pos, y=y_no_interference,
                marker_color=colors_no_int,
                name="No Interference",
                showlegend=False
            ), row=1, col=1)

            # Outcomes with interference (spillover)
            spillover_effect = np.array([np.sum(treatment) / n_units * 5 for _ in range(n_units)])
            y_with_interference = y_no_interference + spillover_effect

            fig.add_trace(go.Bar(
                x=x_pos, y=y_with_interference,
                marker_color=colors_no_int,
                name="With Interference",
                showlegend=False
            ), row=2, col=1)

            fig.update_layout(
                title="SUTVA Violation: Interference Effects",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### 🔧 Dealing with SUTVA Violations
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Design Solutions:**
            - **Cluster randomization**: Randomize groups, not individuals
            - **Geographical separation**: Create buffer zones
            - **Temporal separation**: Stagger treatment timing
            - **Market-level randomization**: Randomize at market level
            """)

        with col2:
            st.markdown("""
            **Analytical Solutions:**
            - **Network analysis**: Model spillover explicitly
            - **Partial population experiments**: Vary treatment intensity
            - **Structural models**: Model equilibrium effects
            - **Bounds analysis**: Provide bounds under different scenarios
            """)

    with tab2:
        st.markdown('<h3 class="subsection-header">🎲 Ignorability (Unconfoundedness)</h3>', unsafe_allow_html=True)

        st.markdown("""
        <div class="important-box">
        <h4>📋 Ignorability Definition</h4>
        <p><strong>Strong Ignorability</strong> requires two conditions:</p>
        <ol>
            <li><strong>Unconfoundedness</strong>: $(Y_0, Y_1) ⊥ T | X$</li>
            <li><strong>Overlap</strong>: $0 < P(T = 1 | X = x) < 1$ for all $x$</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### Understanding Unconfoundedness
        """)

        st.latex(r'''
        (Y_{0i}, Y_{1i}) \perp T_i | X_i
        ''')

        st.markdown("""
        **In words:** Conditional on observed covariates X, treatment assignment is independent of potential outcomes.

        **Alternative formulations:**
        """)

        st.latex(r'''
        \begin{align}
        P(T_i = 1 | Y_{0i}, Y_{1i}, X_i) &= P(T_i = 1 | X_i) \\
        E[Y_{0i} | T_i = 1, X_i] &= E[Y_{0i} | T_i = 0, X_i] \\
        E[Y_{1i} | T_i = 1, X_i] &= E[Y_{1i} | T_i = 0, X_i]
        \end{align}
        ''')

        # Interactive demonstration of confounding
        st.markdown("#### 🎮 Interactive Confounding Demonstration")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Adjust Confounding:**")
            confounder_strength = st.slider("Confounder effect on treatment", 0.0, 2.0, 1.0, 0.1)
            confounder_outcome_effect = st.slider("Confounder effect on outcome", 0, 20, 10)
            include_covariate = st.checkbox("Include covariate in analysis", True)

        with col2:
            np.random.seed(42)
            n = 500

            # Confounder
            confounder = np.random.normal(0, 1, n)

            # Treatment depends on confounder
            logit_p = -0.5 + confounder_strength * confounder
            prob_treatment = 1 / (1 + np.exp(-logit_p))
            treatment = np.random.binomial(1, prob_treatment, n)

            # Outcome depends on treatment and confounder
            true_effect = 8
            outcome = 50 + true_effect * treatment + confounder_outcome_effect * confounder + np.random.normal(0, 3, n)

            # Estimates
            naive_est = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])

            if include_covariate:
                from sklearn.linear_model import LinearRegression

                reg = LinearRegression()
                X_reg = np.column_stack([treatment, confounder])
                reg.fit(X_reg, outcome)
                adjusted_est = reg.coef_[0]
            else:
                adjusted_est = naive_est

            # Visualization
            fig = px.scatter(
                x=confounder, y=outcome, color=treatment.astype(str),
                title=f"Confounding Demonstration<br>Naive Est: {naive_est:.2f} | Adjusted Est: {adjusted_est:.2f} | True: {true_effect}",
                labels={'x': 'Confounder', 'y': 'Outcome', 'color': 'Treatment'}
            )

            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **Bias Analysis:**
            - True Effect: {true_effect}
            - Naive Bias: {naive_est - true_effect:.2f}
            - Adjusted Bias: {adjusted_est - true_effect:.2f}
            """)

        st.markdown("""
        #### Common Violations and Solutions
        """)

        violations_data = {
            "Violation Type": ["Omitted Variable Bias", "Selection Bias", "Reverse Causality", "Measurement Error"],
            "Description": [
                "Important confounders not observed/included",
                "Non-random selection into treatment",
                "Outcome affects treatment assignment",
                "Key variables measured with error"
            ],
            "Example": [
                "Ability affects both education and wages",
                "Sicker patients get more intensive treatment",
                "Expecting benefits affects program participation",
                "Self-reported income, health status"
            ],
            "Solution": [
                "Instrumental variables, fixed effects",
                "Randomization, matching, propensity scores",
                "Instrumental variables, lag structure",
                "Multiple measurements, validation studies"
            ]
        }

        df_violations = pd.DataFrame(violations_data)

        for i, row in df_violations.iterrows():
            with st.expander(f"🚫 {row['Violation Type']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Description:**\n{row['Description']}")
                with col2:
                    st.markdown(f"**Example:**\n{row['Example']}")
                with col3:
                    st.markdown(f"**Solution:**\n{row['Solution']}")

    with tab3:
        st.markdown('<h3 class="subsection-header">🔄 Overlap (Common Support)</h3>', unsafe_allow_html=True)

        st.markdown("""
        <div class="important-box">
        <h4>📋 Overlap Definition</h4>
        <p><strong>Overlap (Common Support)</strong>: For all values of covariates X that occur in the population, 
        there must be positive probability of receiving both treatment and control.</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        0 < P(T = 1 | X = x) < 1 \quad \forall x \in \text{supp}(X)
        ''')

        st.markdown("""
        **Why is overlap important?**
        - Without overlap, we're extrapolating beyond the data
        - No comparable units means no valid counterfactual
        - Identification relies on having both treated and control units at each X value
        """)

        # Overlap visualization
        st.markdown("#### 📊 Overlap Visualization")

        overlap_scenario = st.selectbox(
            "Choose overlap scenario:",
            ["Good Overlap", "Poor Overlap", "No Overlap Region", "Perfect Separation"]
        )

        np.random.seed(42)
        n = 500
        x = np.random.normal(0, 1, n)

        if overlap_scenario == "Good Overlap":
            logit_p = 0.2 * x
        elif overlap_scenario == "Poor Overlap":
            logit_p = 1.5 * x
        elif overlap_scenario == "No Overlap Region":
            logit_p = 3 * x
        else:  # Perfect Separation
            logit_p = 5 * x

        prob_treatment = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, prob_treatment, n)

        col1, col2 = st.columns(2)

        with col1:
            # Propensity score distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=prob_treatment[treatment == 0],
                name='Control',
                opacity=0.7,
                marker_color='red',
                nbinsx=20
            ))

            fig.add_trace(go.Histogram(
                x=prob_treatment[treatment == 1],
                name='Treated',
                opacity=0.7,
                marker_color='blue',
                nbinsx=20
            ))

            fig.update_layout(
                title="Propensity Score Distribution",
                xaxis_title="Propensity Score",
                yaxis_title="Frequency",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Covariate distribution by treatment
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=x[treatment == 0],
                name='Control',
                opacity=0.7,
                marker_color='red',
                nbinsx=20
            ))

            fig.add_trace(go.Histogram(
                x=x[treatment == 1],
                name='Treated',
                opacity=0.7,
                marker_color='blue',
                nbinsx=20
            ))

            fig.update_layout(
                title="Covariate Distribution by Treatment",
                xaxis_title="Covariate X",
                yaxis_title="Frequency",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Overlap quality metrics
        overlap_metrics = {
            "Metric": ["Minimum PS (Treated)", "Maximum PS (Control)", "Normalized Difference", "Overlap Coefficient"],
            "Good Overlap": ["0.05", "0.95", "< 0.25", "> 0.8"],
            "Poor Overlap": ["0.01", "0.99", "0.25 - 0.5", "0.5 - 0.8"],
            "No Overlap": ["< 0.01", "> 0.99", "> 0.5", "< 0.5"]
        }

        df_overlap = pd.DataFrame(overlap_metrics)
        st.markdown("#### 📏 Overlap Quality Metrics")
        st.dataframe(df_overlap, use_container_width=True)

        # Calculate actual metrics
        ps_treated_min = np.min(prob_treatment[treatment == 1])
        ps_control_max = np.max(prob_treatment[treatment == 0])

        # Normalized difference
        mean_diff = np.mean(x[treatment == 1]) - np.mean(x[treatment == 0])
        pooled_sd = np.sqrt((np.var(x[treatment == 1]) + np.var(x[treatment == 0])) / 2)
        norm_diff = mean_diff / pooled_sd

        st.markdown(f"""
        **Current Scenario Metrics:**
        - Minimum PS (Treated): {ps_treated_min:.3f}
        - Maximum PS (Control): {ps_control_max:.3f}
        - Normalized Difference: {norm_diff:.3f}
        """)

        # Solutions for overlap problems
        st.markdown("""
        #### 🔧 Solutions for Overlap Problems
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Design-Based Solutions:**
            - Stratified sampling
            - Oversampling rare groups
            - Targeted data collection
            - Matched sampling
            """)

        with col2:
            st.markdown("""
            **Analysis-Based Solutions:**
            - Trimming extreme propensity scores
            - Weighting adjustments
            - Subclassification
            - Focus on regions of overlap
            """)

    with tab4:
        st.markdown('<h3 class="subsection-header">📊 Testing and Diagnosing Assumptions</h3>', unsafe_allow_html=True)

        test_type = st.selectbox(
            "Choose assumption test:",
            ["Balance Tests", "Overlap Diagnostics", "Placebo Tests", "Sensitivity Analysis"]
        )

        if test_type == "Balance Tests":
            st.markdown("""
            #### Balance Tests for Unconfoundedness

            **Goal:** Check if treated and control groups are similar on observed characteristics
            """)

            st.markdown("""
            **Common Balance Statistics:**
            """)

            st.latex(r'''
            \begin{align}
            \text{Standardized Difference} &= \frac{\bar{X}_1 - \bar{X}_0}{\sqrt{(S_1^2 + S_0^2)/2}} \\
            \text{Variance Ratio} &= \frac{S_1^2}{S_0^2} \\
            \text{t-statistic} &= \frac{\bar{X}_1 - \bar{X}_0}{\sqrt{S_1^2/n_1 + S_0^2/n_0}}
            \end{align}
            ''')

            # Generate imbalanced data
            np.random.seed(42)
            n = 500

            # Covariates with varying degrees of imbalance
            x1 = np.random.normal(0, 1, n)  # Balanced
            x2 = np.random.normal(0, 1, n) + 0.5 * np.random.binomial(1, 0.5, n)  # Moderately imbalanced
            x3 = np.random.normal(0, 1, n) + 1.5 * np.random.binomial(1, 0.5, n)  # Highly imbalanced

            # Treatment assignment
            logit_p = -0.5 + 0.3 * x1 + 0.8 * x2 + 1.2 * x3
            prob_treatment = 1 / (1 + np.exp(-logit_p))
            treatment = np.random.binomial(1, prob_treatment, n)

            # Calculate balance statistics
            covariates = {'X1': x1, 'X2': x2, 'X3': x3}
            balance_results = []

            for name, covar in covariates.items():
                mean_1 = np.mean(covar[treatment == 1])
                mean_0 = np.mean(covar[treatment == 0])
                var_1 = np.var(covar[treatment == 1], ddof=1)
                var_0 = np.var(covar[treatment == 0], ddof=1)

                std_diff = (mean_1 - mean_0) / np.sqrt((var_1 + var_0) / 2)
                var_ratio = var_1 / var_0

                # t-test
                t_stat, p_val = stats.ttest_ind(covar[treatment == 1], covar[treatment == 0])

                balance_results.append({
                    'Covariate': name,
                    'Mean (T=1)': mean_1,
                    'Mean (T=0)': mean_0,
                    'Std Diff': std_diff,
                    'Var Ratio': var_ratio,
                    't-stat': t_stat,
                    'p-value': p_val
                })

            df_balance = pd.DataFrame(balance_results)
            st.dataframe(df_balance.round(3), use_container_width=True)

            # Balance plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_balance['Std Diff'],
                y=df_balance['Covariate'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=np.abs(df_balance['Std Diff']),
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Abs(Std Diff)")
                ),
                text=[f"Std Diff: {x:.3f}" for x in df_balance['Std Diff']],
                textposition="middle right"
            ))

            # Add reference lines
            fig.add_vline(x=0.1, line_dash="dash", line_color="green", annotation_text="Good Balance")
            fig.add_vline(x=-0.1, line_dash="dash", line_color="green")
            fig.add_vline(x=0.25, line_dash="dash", line_color="orange", annotation_text="Acceptable")
            fig.add_vline(x=-0.25, line_dash="dash", line_color="orange")

            fig.update_layout(
                title="Covariate Balance Plot",
                xaxis_title="Standardized Difference",
                yaxis_title="Covariates"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation Guidelines:**
            - |Std Diff| < 0.1: Good balance
            - 0.1 ≤ |Std Diff| < 0.25: Acceptable balance  
            - |Std Diff| ≥ 0.25: Poor balance (concern)
            """)

        elif test_type == "Overlap Diagnostics":
            st.markdown("""
            #### Overlap Diagnostics

            **Visual Diagnostics:**
            """)

            # Generate data with overlap issues
            np.random.seed(42)
            n = 500
            x = np.random.normal(0, 1, n)

            overlap_quality = st.select_slider(
                "Select overlap quality:",
                options=["Excellent", "Good", "Poor", "Very Poor"],
                value="Good"
            )

            overlap_params = {
                "Excellent": 0.5,
                "Good": 1.0,
                "Poor": 2.0,
                "Very Poor": 3.0
            }

            logit_p = overlap_params[overlap_quality] * x
            prob_treatment = 1 / (1 + np.exp(-logit_p))
            treatment = np.random.binomial(1, prob_treatment, n)

            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Propensity Score by Treatment", "Propensity Score Distribution",
                                "Covariate by Treatment", "Common Support Region"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Plot 1: Propensity score by treatment
            fig.add_trace(go.Box(
                y=prob_treatment[treatment == 0],
                name='Control',
                marker_color='red',
                showlegend=False
            ), row=1, col=1)

            fig.add_trace(go.Box(
                y=prob_treatment[treatment == 1],
                name='Treated',
                marker_color='blue',
                showlegend=False
            ), row=1, col=1)

            # Plot 2: Propensity score histograms
            fig.add_trace(go.Histogram(
                x=prob_treatment[treatment == 0],
                name='Control',
                marker_color='red',
                opacity=0.7,
                nbinsx=20,
                showlegend=False
            ), row=1, col=2)

            fig.add_trace(go.Histogram(
                x=prob_treatment[treatment == 1],
                name='Treated',
                marker_color='blue',
                opacity=0.7,
                nbinsx=20,
                showlegend=False
            ), row=1, col=2)

            # Plot 3: Covariate by treatment
            fig.add_trace(go.Scatter(
                x=x[treatment == 0],
                y=prob_treatment[treatment == 0],
                mode='markers',
                name='Control',
                marker=dict(color='red', opacity=0.6),
                showlegend=False
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=x[treatment == 1],
                y=prob_treatment[treatment == 1],
                mode='markers',
                name='Treated',
                marker=dict(color='blue', opacity=0.6),
                showlegend=False
            ), row=2, col=1)

            # Plot 4: Common support
            ps_bins = np.linspace(0, 1, 21)
            support_mask = np.zeros(len(ps_bins) - 1, dtype=bool)

            for i in range(len(ps_bins) - 1):
                in_bin = (prob_treatment >= ps_bins[i]) & (prob_treatment < ps_bins[i + 1])
                has_treated = np.any(treatment[in_bin] == 1)
                has_control = np.any(treatment[in_bin] == 0)
                support_mask[i] = has_treated and has_control

            fig.add_trace(go.Bar(
                x=(ps_bins[:-1] + ps_bins[1:]) / 2,
                y=support_mask.astype(int),
                name='Common Support',
                marker_color=['green' if s else 'red' for s in support_mask],
                showlegend=False
            ), row=2, col=2)

            fig.update_layout(height=600, title="Overlap Diagnostic Plots")
            st.plotly_chart(fig, use_container_width=True)

            # Overlap metrics
            min_ps_treated = np.min(prob_treatment[treatment == 1])
            max_ps_control = np.max(prob_treatment[treatment == 0])
            overlap_region = (min_ps_treated <= prob_treatment) & (prob_treatment <= max_ps_control)
            overlap_prop = np.mean(overlap_region)

            st.markdown(f"""
            **Overlap Metrics:**
            - Minimum PS (Treated): {min_ps_treated:.3f}
            - Maximum PS (Control): {max_ps_control:.3f}
            - Proportion in Overlap: {overlap_prop:.3f}
            - Quality: {overlap_quality}
            """)

        elif test_type == "Placebo Tests":
            st.markdown("""
            #### Placebo Tests

            **Idea:** Use pre-treatment outcomes or unlikely-to-be-affected outcomes as "placebo" tests
            """)

            st.latex(r'''
            H_0: \text{No effect on placebo outcome} \\
            H_1: \text{Effect on placebo outcome (suggests confounding)}
            ''')

            # Simulate placebo test
            np.random.seed(42)
            n = 300

            # Confounder affects treatment and both outcomes
            confounder = np.random.normal(0, 1, n)

            # Treatment assignment
            logit_p = 0.5 * confounder
            prob_treatment = 1 / (1 + np.exp(-logit_p))
            treatment = np.random.binomial(1, prob_treatment, n)

            # True outcome (affected by treatment)
            true_outcome = 50 + 8 * treatment + 5 * confounder + np.random.normal(0, 3, n)

            # Placebo outcomes
            placebo_unaffected = 30 + 3 * confounder + np.random.normal(0, 2, n)  # Not affected by treatment
            placebo_pretreatment = 40 + 4 * confounder + np.random.normal(0, 2, n)  # Pre-treatment outcome

            # Analysis
            placebo_tests = {
                'Outcome': ['True Outcome', 'Placebo (Unaffected)', 'Placebo (Pre-treatment)'],
                'Estimated Effect': [],
                'p-value': [],
                'Interpretation': []
            }

            outcomes = [true_outcome, placebo_unaffected, placebo_pretreatment]
            interpretations = ['Expected effect', 'Should be zero', 'Should be zero']

            for i, outcome in enumerate(outcomes):
                t_stat, p_val = stats.ttest_ind(outcome[treatment == 1], outcome[treatment == 0])
                effect = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])

                placebo_tests['Estimated Effect'].append(effect)
                placebo_tests['p-value'].append(p_val)
                placebo_tests['Interpretation'].append(interpretations[i])

            df_placebo = pd.DataFrame(placebo_tests)
            st.dataframe(df_placebo.round(3), use_container_width=True)

            # Visualization
            fig = go.Figure()

            for i, (outcome_name, outcome) in enumerate(zip(placebo_tests['Outcome'], outcomes)):
                fig.add_trace(go.Box(
                    y=outcome[treatment == 0],
                    name=f'{outcome_name} (Control)',
                    marker_color='red',
                    opacity=0.7,
                    offsetgroup=i,
                    boxpoints='outliers'
                ))

                fig.add_trace(go.Box(
                    y=outcome[treatment == 1],
                    name=f'{outcome_name} (Treated)',
                    marker_color='blue',
                    opacity=0.7,
                    offsetgroup=i,
                    boxpoints='outliers'
                ))

            fig.update_layout(
                title="Placebo Tests: Outcome Distributions",
                yaxis_title="Outcome Value",
                boxmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - If placebo tests show significant effects, it suggests confounding
            - Pre-treatment outcomes should not be affected by treatment
            - Unrelated outcomes should not show treatment effects
            """)

        elif test_type == "Sensitivity Analysis":
            st.markdown("""
            #### Sensitivity Analysis for Unobserved Confounding

            **Question:** How much unobserved confounding would be needed to explain away the estimated effect?
            """)

            # Cornfield's inequality and bounds
            st.markdown("""
            **Cornfield's Inequality (for binary outcomes):**
            """)

            st.latex(r'''
            \text{If RR} > \gamma, \text{ then } \gamma < \frac{p(E|D,\overline{U})}{p(E|\overline{D},U)} \cdot \frac{p(E|\overline{D},\overline{U})}{p(E|D,U)}
            ''')

            # Simulation for sensitivity analysis
            st.markdown("#### 🎮 Interactive Sensitivity Analysis")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Confounding Parameters:**")
                gamma_td = st.slider("Γ_TD: Confounder → Treatment", 1.0, 5.0, 2.0, 0.1)
                gamma_yd = st.slider("Γ_YD: Confounder → Outcome", 1.0, 5.0, 2.0, 0.1)
                confounder_prev = st.slider("Confounder Prevalence", 0.1, 0.9, 0.5, 0.05)

            with col2:
                # Simulate sensitivity analysis
                np.random.seed(42)
                n = 1000

                # Unobserved confounder
                u = np.random.binomial(1, confounder_prev, n)

                # Treatment probability with confounding
                logit_p_base = -0.5
                logit_p_confounded = logit_p_base + np.log(gamma_td) * u
                prob_treatment = 1 / (1 + np.exp(-logit_p_confounded))
                treatment = np.random.binomial(1, prob_treatment, n)

                # Outcome with confounding
                outcome_prob_base = 0.3
                logit_y_base = np.log(outcome_prob_base / (1 - outcome_prob_base))
                true_log_or = np.log(1.5)  # True treatment effect (log odds ratio)
                logit_y = logit_y_base + true_log_or * treatment + np.log(gamma_yd) * u
                prob_outcome = 1 / (1 + np.exp(-logit_y))
                outcome = np.random.binomial(1, prob_outcome, n)

                # Estimate effect ignoring confounder
                from statsmodels.stats.contingency_tables import mcnemar

                # 2x2 table
                tab = pd.crosstab(treatment, outcome, margins=True)

                # Odds ratio
                if tab.iloc[0, 0] * tab.iloc[1, 1] != 0:
                    obs_or = (tab.iloc[1, 1] * tab.iloc[0, 0]) / (tab.iloc[1, 0] * tab.iloc[0, 1])
                    obs_log_or = np.log(obs_or)
                else:
                    obs_or = np.inf
                    obs_log_or = np.inf

                # Display results
                st.markdown(f"""
                **Sensitivity Results:**
                - True Log OR: {true_log_or:.3f}
                - Observed Log OR: {obs_log_or:.3f}
                - Bias: {obs_log_or - true_log_or:.3f}
                - Bias Factor: {np.exp(obs_log_or - true_log_or):.3f}
                """)

                # Create 2x2 table visualization
                fig = go.Figure(data=go.Heatmap(
                    z=tab.iloc[:2, :2].values,
                    x=['No Outcome', 'Outcome'],
                    y=['No Treatment', 'Treatment'],
                    text=tab.iloc[:2, :2].values,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorscale='Blues'
                ))

                fig.update_layout(
                    title=f"2×2 Contingency Table<br>Observed OR = {obs_or:.2f}",
                    xaxis_title="Outcome",
                    yaxis_title="Treatment"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Sensitivity contour plot
            st.markdown("#### 📊 Sensitivity Contour Plot")

            gamma_td_range = np.linspace(1, 4, 20)
            gamma_yd_range = np.linspace(1, 4, 20)

            # Create grid for contour plot
            GTD, GYD = np.meshgrid(gamma_td_range, gamma_yd_range)

            # Simplified bias calculation
            bias_factor = GTD * GYD * confounder_prev * (1 - confounder_prev)

            fig = go.Figure(data=go.Contour(
                z=bias_factor,
                x=gamma_td_range,
                y=gamma_yd_range,
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                colorbar=dict(title="Bias Factor")
            ))

            fig.update_layout(
                title="Sensitivity Analysis: Bias Factor Contours",
                xaxis_title="Γ_TD (Confounder → Treatment)",
                yaxis_title="Γ_YD (Confounder → Outcome)"
            )

            st.plotly_chart(fig, use_container_width=True)

# Section 6: Applications & Examples
elif selected_section == "💼 Applications & Examples":
    st.markdown('<h2 class="section-header">💼 Real-World Applications & Examples</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Classic Examples", "🏥 Healthcare", "📚 Education", "💰 Economics"])

    with tab1:
        st.markdown('<h3 class="subsection-header">📊 Classic RCM Examples</h3>', unsafe_allow_html=True)

        example_choice = st.selectbox(
            "Choose a classic example:",
            ["Job Training Programs", "Medical Treatment Effects", "Educational Interventions"]
        )

        if example_choice == "Job Training Programs":
            st.markdown("""
            <div class="example-box">
            <h4>🎯 Example: National Supported Work Demonstration (NSW)</h4>
            <p><strong>Research Question:</strong> What is the effect of job training on earnings?</p>
            <p><strong>Treatment:</strong> Job training program participation</p>
            <p><strong>Outcome:</strong> Post-program earnings</p>
            </div>
            """, unsafe_allow_html=True)

            # Simulate NSW-like data
            np.random.seed(42)
            n_experimental = 400
            n_observational = 2000

            # Experimental data (randomized)
            treatment_exp = np.random.binomial(1, 0.5, n_experimental)
            # Baseline characteristics
            age_exp = np.random.normal(35, 8, n_experimental)
            education_exp = np.random.normal(10, 2, n_experimental)
            pre_earnings_exp = np.random.exponential(2000, n_experimental)

            # Earnings with true treatment effect
            true_effect = 1500
            post_earnings_exp = (5000 + 500 * (age_exp - 35) + 200 * education_exp +
                                 0.3 * pre_earnings_exp + true_effect * treatment_exp +
                                 np.random.normal(0, 1000, n_experimental))

            # Observational data (selection bias)
            # More motivated/able people select into treatment
            ability = np.random.normal(0, 1, n_observational)
            prob_treatment_obs = 1 / (1 + np.exp(-0.5 - 0.8 * ability))
            treatment_obs = np.random.binomial(1, prob_treatment_obs, n_observational)

            age_obs = np.random.normal(35 + 2 * ability, 8, n_observational)
            education_obs = np.random.normal(10 + ability, 2, n_observational)
            pre_earnings_obs = np.random.exponential(2000 * (1 + 0.3 * ability), n_observational)

            post_earnings_obs = (5000 + 500 * (age_obs - 35) + 200 * education_obs +
                                 0.3 * pre_earnings_obs + true_effect * treatment_obs +
                                 1000 * ability + np.random.normal(0, 1000, n_observational))

            # Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎲 Experimental Results (Gold Standard)")

                ate_exp = np.mean(post_earnings_exp[treatment_exp == 1]) - np.mean(
                    post_earnings_exp[treatment_exp == 0])
                se_exp = np.sqrt(np.var(post_earnings_exp[treatment_exp == 1]) / np.sum(treatment_exp) +
                                 np.var(post_earnings_exp[treatment_exp == 0]) / np.sum(1 - treatment_exp))

                st.markdown(f"""
                - **Sample Size:** {n_experimental}
                - **ATE Estimate:** ${ate_exp:,.0f}
                - **Standard Error:** ${se_exp:,.0f}
                - **95% CI:** [${ate_exp - 1.96 * se_exp:,.0f}, ${ate_exp + 1.96 * se_exp:,.0f}]
                - **True Effect:** ${true_effect:,.0f}
                """)

                # Experimental data visualization
                fig = go.Figure()

                fig.add_trace(go.Box(
                    y=post_earnings_exp[treatment_exp == 0],
                    name='Control',
                    marker_color='red',
                    boxpoints='outliers'
                ))

                fig.add_trace(go.Box(
                    y=post_earnings_exp[treatment_exp == 1],
                    name='Treatment',
                    marker_color='blue',
                    boxpoints='outliers'
                ))

                fig.update_layout(
                    title="Experimental Data: Post-Training Earnings",
                    yaxis_title="Annual Earnings ($)",
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📊 Observational Results (Biased)")

                ate_obs = np.mean(post_earnings_obs[treatment_obs == 1]) - np.mean(
                    post_earnings_obs[treatment_obs == 0])
                se_obs = np.sqrt(np.var(post_earnings_obs[treatment_obs == 1]) / np.sum(treatment_obs) +
                                 np.var(post_earnings_obs[treatment_obs == 0]) / np.sum(1 - treatment_obs))

                st.markdown(f"""
                - **Sample Size:** {n_observational}
                - **ATE Estimate:** ${ate_obs:,.0f}
                - **Standard Error:** ${se_obs:,.0f}
                - **95% CI:** [${ate_obs - 1.96 * se_obs:,.0f}, ${ate_obs + 1.96 * se_obs:,.0f}]
                - **Selection Bias:** ${ate_obs - true_effect:,.0f}
                """)

                # Observational data visualization
                fig = go.Figure()

                fig.add_trace(go.Box(
                    y=post_earnings_obs[treatment_obs == 0],
                    name='Control',
                    marker_color='red',
                    boxpoints='outliers'
                ))

                fig.add_trace(go.Box(
                    y=post_earnings_obs[treatment_obs == 1],
                    name='Treatment',
                    marker_color='blue',
                    boxpoints='outliers'
                ))

                fig.update_layout(
                    title="Observational Data: Post-Training Earnings",
                    yaxis_title="Annual Earnings ($)",
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            # Covariate balance comparison
            st.markdown("#### ⚖️ Covariate Balance Comparison")

            covariates_exp = pd.DataFrame({
                'Age': age_exp,
                'Education': education_exp,
                'Pre_Earnings': pre_earnings_exp,
                'Treatment': treatment_exp
            })

            covariates_obs = pd.DataFrame({
                'Age': age_obs,
                'Education': education_obs,
                'Pre_Earnings': pre_earnings_obs,
                'Treatment': treatment_obs
            })

            balance_comparison = []

            for covar in ['Age', 'Education', 'Pre_Earnings']:
                # Experimental balance
                std_diff_exp = ((covariates_exp[covariates_exp['Treatment'] == 1][covar].mean() -
                                 covariates_exp[covariates_exp['Treatment'] == 0][covar].mean()) /
                                np.sqrt((covariates_exp[covariates_exp['Treatment'] == 1][covar].var() +
                                         covariates_exp[covariates_exp['Treatment'] == 0][covar].var()) / 2))

                # Observational balance
                std_diff_obs = ((covariates_obs[covariates_obs['Treatment'] == 1][covar].mean() -
                                 covariates_obs[covariates_obs['Treatment'] == 0][covar].mean()) /
                                np.sqrt((covariates_obs[covariates_obs['Treatment'] == 1][covar].var() +
                                         covariates_obs[covariates_obs['Treatment'] == 0][covar].var()) / 2))

                balance_comparison.append({
                    'Covariate': covar,
                    'Experimental Std Diff': std_diff_exp,
                    'Observational Std Diff': std_diff_obs
                })

            df_balance_comp = pd.DataFrame(balance_comparison)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_balance_comp['Experimental Std Diff'],
                y=df_balance_comp['Covariate'],
                mode='markers',
                marker=dict(color='blue', size=12),
                name='Experimental'
            ))

            fig.add_trace(go.Scatter(
                x=df_balance_comp['Observational Std Diff'],
                y=df_balance_comp['Covariate'],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Observational'
            ))

            # Add reference lines
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
            fig.add_vline(x=0.1, line_dash="dash", line_color="green")
            fig.add_vline(x=-0.1, line_dash="dash", line_color="green")
            fig.add_vline(x=0.25, line_dash="dash", line_color="orange")
            fig.add_vline(x=-0.25, line_dash="dash", line_color="orange")

            fig.update_layout(
                title="Covariate Balance: Experimental vs Observational",
                xaxis_title="Standardized Difference",
                yaxis_title="Covariates"
            )

            st.plotly_chart(fig, use_container_width=True)

        elif example_choice == "Medical Treatment Effects":
            st.markdown("""
            <div class="example-box">
            <h4>🏥 Example: Drug Effectiveness Study</h4>
            <p><strong>Research Question:</strong> Does the new drug reduce symptoms better than placebo?</p>
            <p><strong>Treatment:</strong> New drug vs placebo</p>
            <p><strong>Outcome:</strong> Symptom severity score (0-100)</p>
            </div>
            """, unsafe_allow_html=True)

            # Medical study simulation
            np.random.seed(42)
            n_patients = 300

            # Patient characteristics
            age = np.random.normal(55, 15, n_patients)
            severity_baseline = np.random.normal(70, 10, n_patients)

            # Randomized treatment assignment
            treatment = np.random.binomial(1, 0.5, n_patients)

            # Treatment effect with heterogeneity
            # Effect varies by baseline severity
            individual_effects = 15 + 0.2 * (severity_baseline - 70) + np.random.normal(0, 5, n_patients)

            # Post-treatment outcomes
            outcome = (severity_baseline - individual_effects * treatment +
                       np.random.normal(0, 8, n_patients))

            # Ensure outcomes are in valid range
            outcome = np.clip(outcome, 0, 100)

            col1, col2 = st.columns(2)

            with col1:
                # Treatment effect analysis
                ate = np.mean(severity_baseline[treatment == 0] - outcome[treatment == 0]) - \
                      np.mean(severity_baseline[treatment == 1] - outcome[treatment == 1])

                # Alternative: direct comparison of outcomes (flipped because lower is better)
                ate_direct = np.mean(outcome[treatment == 0]) - np.mean(outcome[treatment == 1])

                st.markdown(f"""
                #### 📊 Treatment Effect Analysis

                **Average Treatment Effect:**
                - Symptom Reduction: {ate_direct:.1f} points
                - Standard Error: {np.sqrt(np.var(outcome[treatment == 1]) / np.sum(treatment) + np.var(outcome[treatment == 0]) / np.sum(1 - treatment)):.1f}
                - Effect Size: {ate_direct / np.std(outcome):.2f} (Cohen's d)
                """)

                # Statistical test
                from scipy.stats import ttest_ind

                t_stat, p_val = ttest_ind(outcome[treatment == 1], outcome[treatment == 0])

                st.markdown(f"""
                **Statistical Significance:**
                - t-statistic: {t_stat:.3f}
                - p-value: {p_val:.4f}
                - Significant: {'Yes' if p_val < 0.05 else 'No'} (α = 0.05)
                """)

            with col2:
                # Before/After comparison
                fig = go.Figure()

                # Control group
                fig.add_trace(go.Scatter(
                    x=['Baseline', 'Post-Treatment'],
                    y=[np.mean(severity_baseline[treatment == 0]), np.mean(outcome[treatment == 0])],
                    mode='lines+markers',
                    name='Control (Placebo)',
                    line=dict(color='red', width=3),
                    marker=dict(size=10)
                ))

                # Treatment group
                fig.add_trace(go.Scatter(
                    x=['Baseline', 'Post-Treatment'],
                    y=[np.mean(severity_baseline[treatment == 1]), np.mean(outcome[treatment == 1])],
                    mode='lines+markers',
                    name='Treatment (Drug)',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))

                fig.update_layout(
                    title="Mean Symptom Severity Over Time",
                    yaxis_title="Symptom Severity (0-100)",
                    xaxis_title="Time Point"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Heterogeneous treatment effects
            st.markdown("#### 🎯 Heterogeneous Treatment Effects by Baseline Severity")

            # Divide into severity terciles
            severity_terciles = np.percentile(severity_baseline, [33.33, 66.67])
            severity_groups = np.digitize(severity_baseline, severity_terciles)

            hte_results = []
            for group in range(3):
                mask = severity_groups == group
                if np.sum(mask & (treatment == 1)) > 0 and np.sum(mask & (treatment == 0)) > 0:
                    group_ate = np.mean(outcome[mask & (treatment == 0)]) - np.mean(outcome[mask & (treatment == 1)])
                    group_labels = ['Low Severity', 'Medium Severity', 'High Severity'][group]
                    hte_results.append({
                        'Group': group_labels,
                        'Treatment Effect': group_ate,
                        'Sample Size': np.sum(mask)
                    })

            df_hte = pd.DataFrame(hte_results)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(df_hte, use_container_width=True)

            with col2:
                fig = go.Figure(data=[
                    go.Bar(x=df_hte['Group'], y=df_hte['Treatment Effect'],
                           marker_color=['lightcoral', 'gold', 'lightblue'])
                ])

                fig.update_layout(
                    title="Treatment Effects by Baseline Severity",
                    yaxis_title="Treatment Effect (Symptom Reduction)",
                    xaxis_title="Baseline Severity Group"
                )

                st.plotly_chart(fig, use_container_width=True)

        elif example_choice == "Educational Interventions":
            st.markdown("""
            <div class="example-box">
            <h4>📚 Example: Class Size Reduction (Tennessee STAR)</h4>
            <p><strong>Research Question:</strong> Does reducing class size improve student achievement?</p>
            <p><strong>Treatment:</strong> Small class (13-17 students) vs regular class (22-25 students)</p>
            <p><strong>Outcome:</strong> Standardized test scores</p>
            </div>
            """, unsafe_allow_html=True)

            # STAR experiment simulation
            np.random.seed(42)
            n_students = 600
            n_schools = 30

            # Student and school characteristics
            school_id = np.repeat(range(n_schools), n_students // n_schools)
            school_quality = np.random.normal(0, 1, n_schools)[school_id]

            # Student characteristics
            ses = np.random.normal(0, 1, n_students)  # Socioeconomic status
            prior_achievement = np.random.normal(100 + 10 * ses + 5 * school_quality, 15, n_students)

            # Within-school randomization to small vs regular classes
            treatment = np.zeros(n_students)
            for school in range(n_schools):
                school_mask = school_id == school
                school_students = np.sum(school_mask)
                treatment[school_mask] = np.random.binomial(1, 0.5, school_students)

            # Treatment effect: small classes help more disadvantaged students
            treatment_effect = 8 - 3 * ses  # Larger effect for lower SES students

            # Post-treatment test scores
            test_scores = (prior_achievement + treatment_effect * treatment +
                           np.random.normal(0, 10, n_students))

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Overall Treatment Effects")

                ate_overall = np.mean(test_scores[treatment == 1]) - np.mean(test_scores[treatment == 0])

                # Clustered standard errors (simplified)
                school_effects = []
                for school in range(n_schools):
                    school_mask = school_id == school
                    if np.sum(school_mask & (treatment == 1)) > 0 and np.sum(school_mask & (treatment == 0)) > 0:
                        school_ate = (np.mean(test_scores[school_mask & (treatment == 1)]) -
                                      np.mean(test_scores[school_mask & (treatment == 0)]))
                        school_effects.append(school_ate)

                se_clustered = np.std(school_effects) / np.sqrt(len(school_effects))

                st.markdown(f"""
                **Class Size Reduction Effects:**
                - Average Treatment Effect: {ate_overall:.2f} points
                - Clustered SE: {se_clustered:.2f}
                - 95% CI: [{ate_overall - 1.96 * se_clustered:.2f}, {ate_overall + 1.96 * se_clustered:.2f}]
                """)

                # Distribution comparison
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=test_scores[treatment == 0],
                    name='Regular Class',
                    opacity=0.7,
                    marker_color='red',
                    nbinsx=25
                ))

                fig.add_trace(go.Histogram(
                    x=test_scores[treatment == 1],
                    name='Small Class',
                    opacity=0.7,
                    marker_color='blue',
                    nbinsx=25
                ))

                fig.update_layout(
                    title="Test Score Distributions",
                    xaxis_title="Test Score",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 🎯 Effects by Socioeconomic Status")

                # Divide by SES quartiles
                ses_quartiles = np.percentile(ses, [25, 50, 75])
                ses_groups = np.digitize(ses, ses_quartiles)

                ses_effects = []
                for group in range(4):
                    mask = ses_groups == group
                    if np.sum(mask & (treatment == 1)) > 10 and np.sum(mask & (treatment == 0)) > 10:
                        group_ate = np.mean(test_scores[mask & (treatment == 1)]) - np.mean(
                            test_scores[mask & (treatment == 0)])
                        group_labels = ['Bottom 25%', '25-50%', '50-75%', 'Top 25%'][group]
                        ses_effects.append({
                            'SES Group': group_labels,
                            'Treatment Effect': group_ate,
                            'Sample Size': np.sum(mask)
                        })

                df_ses = pd.DataFrame(ses_effects)
                st.dataframe(df_ses.round(2), use_container_width=True)

                # Visualization
                fig = go.Figure()

                colors = ['darkred', 'orange', 'lightblue', 'darkblue']
                for i, row in df_ses.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['SES Group']],
                        y=[row['Treatment Effect']],
                        name=row['SES Group'],
                        marker_color=colors[i],
                        showlegend=False
                    ))

                fig.update_layout(
                    title="Treatment Effects by SES Quartile",
                    yaxis_title="Treatment Effect (Test Score Points)",
                    xaxis_title="Socioeconomic Status"
                )

                st.plotly_chart(fig, use_container_width=True)

            # School-level analysis
            st.markdown("#### 🏫 School-Level Heterogeneity")

            school_results = []
            for school in range(min(15, n_schools)):  # Show first 15 schools
                school_mask = school_id == school
                if np.sum(school_mask & (treatment == 1)) > 0 and np.sum(school_mask & (treatment == 0)) > 0:
                    school_ate = (np.mean(test_scores[school_mask & (treatment == 1)]) -
                                  np.mean(test_scores[school_mask & (treatment == 0)]))
                    school_quality_val = school_quality[school_mask][0]
                    school_results.append({
                        'School': f'School {school + 1}',
                        'Treatment Effect': school_ate,
                        'School Quality': school_quality_val
                    })

            df_schools = pd.DataFrame(school_results)

            fig = px.scatter(
                df_schools,
                x='School Quality',
                y='Treatment Effect',
                hover_data=['School'],
                title="School-Level Treatment Effects vs School Quality",
                labels={'School Quality': 'School Quality Index', 'Treatment Effect': 'Treatment Effect (Points)'}
            )

            fig.add_hline(y=0, line_dash="dash", line_color="red",
                          annotation_text="No Effect")

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown('<h3 class="subsection-header">🏥 Healthcare Applications</h3>', unsafe_allow_html=True)

            healthcare_example = st.selectbox(
                "Choose a healthcare application:",
                ["Clinical Trials Design", "Observational Studies", "Health Policy Evaluation"]
            )

            if healthcare_example == "Clinical Trials Design":
                st.markdown("""
                    <div class="example-box">
                    <h4>💊 Clinical Trial: Cancer Treatment Efficacy</h4>
                    <p>Using RCM principles to design and analyze a randomized controlled trial</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Clinical trial simulation
                np.random.seed(123)
                n_patients = 400

                # Patient characteristics
                age = np.random.normal(65, 12, n_patients)
                stage = np.random.choice([1, 2, 3, 4], n_patients, p=[0.3, 0.4, 0.2, 0.1])
                biomarker = np.random.normal(50, 20, n_patients)

                # Stratified randomization by cancer stage
                treatment = np.zeros(n_patients, dtype=int)
                for s in [1, 2, 3, 4]:
                    stage_mask = stage == s
                    n_stage = np.sum(stage_mask)
                    treatment[stage_mask] = np.random.binomial(1, 0.5, n_stage)

                # Survival time (months) - Weibull distribution
                # Treatment effect varies by stage and biomarker
                baseline_survival = 12 + 2 * (65 - age) - 3 * stage + 0.1 * biomarker
                treatment_effect = 6 * (1 + 0.02 * biomarker) * (5 - stage) / 4

                survival_time = np.random.weibull(2, n_patients) * np.exp(
                    baseline_survival + treatment_effect * treatment)
                survival_time = np.clip(survival_time, 1, 60)  # 1-60 months

                # Event indicator (1 = death, 0 = censored)
                event = np.random.binomial(1, 0.7, n_patients)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Trial Characteristics")

                    # Balance table
                    balance_data = []
                    for var, values in [('Age', age), ('Stage', stage), ('Biomarker', biomarker)]:
                        control_mean = np.mean(values[treatment == 0])
                        treated_mean = np.mean(values[treatment == 1])
                        std_diff = abs(treated_mean - control_mean) / np.sqrt(
                            (np.var(values[treatment == 0]) + np.var(values[treatment == 1])) / 2)

                        balance_data.append({
                            'Variable': var,
                            'Control Mean': f"{control_mean:.1f}",
                            'Treatment Mean': f"{treated_mean:.1f}",
                            'Std Diff': f"{std_diff:.3f}"
                        })

                    df_balance = pd.DataFrame(balance_data)
                    st.dataframe(df_balance, use_container_width=True)

                    # Sample sizes
                    st.markdown(f"""
                        **Sample Allocation:**
                        - Control Group: {np.sum(treatment == 0)} patients
                        - Treatment Group: {np.sum(treatment == 1)} patients
                        - Total: {n_patients} patients
                        """)

                with col2:
                    # Kaplan-Meier curves simulation
                    fig = go.Figure()

                    # Time points for survival curves
                    time_points = np.linspace(0, 60, 100)

                    for group, color, name in [(0, 'red', 'Control'), (1, 'blue', 'Treatment')]:
                        group_survival = survival_time[treatment == group]
                        group_events = event[treatment == group]

                        # Simple survival curve estimation
                        survival_probs = []
                        for t in time_points:
                            at_risk = np.sum(group_survival >= t)
                            if at_risk > 0:
                                prob = at_risk / len(group_survival)
                            else:
                                prob = 0
                            survival_probs.append(prob)

                        fig.add_trace(go.Scatter(
                            x=time_points,
                            y=survival_probs,
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=3)
                        ))

                    fig.update_layout(
                        title="Kaplan-Meier Survival Curves",
                        xaxis_title="Time (months)",
                        yaxis_title="Survival Probability",
                        yaxis=dict(range=[0, 1])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Treatment effect by subgroup
                st.markdown("#### 🎯 Subgroup Analysis")

                subgroup_results = []

                # By cancer stage
                for s in [1, 2, 3, 4]:
                    mask = stage == s
                    if np.sum(mask) > 20:  # Sufficient sample size
                        median_control = np.median(survival_time[mask & (treatment == 0)])
                        median_treated = np.median(survival_time[mask & (treatment == 1)])
                        hazard_ratio = median_control / median_treated if median_treated > 0 else np.nan

                        subgroup_results.append({
                            'Subgroup': f'Stage {s}',
                            'Control Median': f"{median_control:.1f}",
                            'Treatment Median': f"{median_treated:.1f}",
                            'Hazard Ratio': f"{hazard_ratio:.2f}" if not np.isnan(hazard_ratio) else "N/A",
                            'Sample Size': np.sum(mask)
                        })

                df_subgroups = pd.DataFrame(subgroup_results)
                st.dataframe(df_subgroups, use_container_width=True)

            elif healthcare_example == "Observational Studies":
                st.markdown("""
                    <div class="warning-box">
                    <h4>⚠️ Observational Healthcare Data Challenges</h4>
                    <p>In observational healthcare studies, treatment assignment is not random, leading to potential confounding.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Observational study simulation
                np.random.seed(456)
                n_patients = 1000

                # Patient characteristics (confounders)
                age = np.random.normal(70, 15, n_patients)
                comorbidity_score = np.random.poisson(2, n_patients)
                insurance = np.random.choice([0, 1], n_patients, p=[0.3, 0.7])  # 0=uninsured, 1=insured

                # Treatment selection (non-random) - healthier, insured patients more likely to get treatment
                logit_p = -2 + 0.05 * (70 - age) + 0.3 * insurance - 0.2 * comorbidity_score
                prob_treatment = 1 / (1 + np.exp(-logit_p))
                treatment = np.random.binomial(1, prob_treatment, n_patients)

                # Outcome (length of stay) - true treatment effect is 2 days reduction
                true_effect = -2
                length_of_stay = (8 + 0.1 * age + 0.5 * comorbidity_score - 1 * insurance +
                                  true_effect * treatment + np.random.normal(0, 2, n_patients))
                length_of_stay = np.clip(length_of_stay, 1, 30)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Confounding Demonstration")

                    # Naive comparison
                    naive_effect = np.mean(length_of_stay[treatment == 1]) - np.mean(length_of_stay[treatment == 0])

                    st.markdown(f"""
                        **Naive Analysis (Biased):**
                        - Treatment Effect: {naive_effect:.2f} days
                        - True Effect: {true_effect:.2f} days
                        - Bias: {naive_effect - true_effect:.2f} days
                        """)

                    # Show confounding
                    st.markdown("**Evidence of Confounding:**")
                    for var, values, name in [(age, 'Age'), (comorbidity_score, 'Comorbidities'),
                                              (insurance, 'Insurance')]:
                        treated_mean = np.mean(values[treatment == 1])
                        control_mean = np.mean(values[treatment == 0])
                        st.markdown(f"- {name}: Treated={treated_mean:.1f}, Control={control_mean:.1f}")

                with col2:
                    # Propensity score adjustment (simplified)
                    from sklearn.linear_model import LogisticRegression

                    # Estimate propensity scores
                    X = np.column_stack([age, comorbidity_score, insurance])
                    ps_model = LogisticRegression()
                    ps_model.fit(X, treatment)
                    propensity_scores = ps_model.predict_proba(X)[:, 1]

                    # IPW estimator
                    weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
                    weights = np.clip(weights, 0, 10)  # Trim extreme weights

                    ipw_effect = (np.average(length_of_stay[treatment == 1], weights=weights[treatment == 1]) -
                                  np.average(length_of_stay[treatment == 0], weights=weights[treatment == 0]))

                    st.markdown(f"""
                        #### 🔧 Propensity Score Adjustment

                        **IPW Estimator:**
                        - Adjusted Effect: {ipw_effect:.2f} days
                        - Bias Reduction: {abs(naive_effect - true_effect) - abs(ipw_effect - true_effect):.2f} days
                        """)

                    # Propensity score distribution
                    fig = go.Figure()

                    fig.add_trace(go.Histogram(
                        x=propensity_scores[treatment == 0],
                        name='Control',
                        opacity=0.7,
                        marker_color='red',
                        nbinsx=20
                    ))

                    fig.add_trace(go.Histogram(
                        x=propensity_scores[treatment == 1],
                        name='Treated',
                        opacity=0.7,
                        marker_color='blue',
                        nbinsx=20
                    ))

                    fig.update_layout(
                        title="Propensity Score Distributions",
                        xaxis_title="Propensity Score",
                        yaxis_title="Frequency",
                        barmode='overlay'
                    )

                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown('<h3 class="subsection-header">📚 Education Policy Applications</h3>', unsafe_allow_html=True)

            education_example = st.selectbox(
                "Choose an education application:",
                ["School Voucher Programs", "Teacher Training Effects", "Technology in Classrooms"]
            )

            if education_example == "School Voucher Programs":
                st.markdown("""
                    <div class="example-box">
                    <h4>🎓 School Choice and Student Achievement</h4>
                    <p>Evaluating the causal effect of school vouchers on student outcomes using lottery-based assignment</p>
                    </div>
                    """, unsafe_allow_html=True)

                # School voucher simulation
                np.random.seed(789)
                n_students = 800

                # Student characteristics
                family_income = np.random.lognormal(10, 0.8, n_students)  # Family income
                parental_education = np.random.normal(12, 3, n_students)  # Years of education
                baseline_achievement = np.random.normal(500 + 0.02 * family_income + 5 * parental_education, 50,
                                                        n_students)

                # Lottery-based voucher assignment (randomized within income strata)
                income_quartiles = np.percentile(family_income, [25, 50, 75])
                income_groups = np.digitize(family_income, income_quartiles)

                voucher = np.zeros(n_students, dtype=int)
                for group in range(4):
                    group_mask = income_groups == group
                    n_group = np.sum(group_mask)
                    voucher[group_mask] = np.random.binomial(1, 0.5, n_group)

                # Treatment effects (vouchers help low-income students more)
                income_percentile = stats.rankdata(family_income) / len(family_income)
                treatment_effect = 25 * (1 - income_percentile) + np.random.normal(0, 10, n_students)

                # Post-treatment achievement
                post_achievement = baseline_achievement + treatment_effect * voucher + np.random.normal(0, 30,
                                                                                                        n_students)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Overall Program Effects")

                    ate_voucher = np.mean(post_achievement[voucher == 1]) - np.mean(post_achievement[voucher == 0])

                    # Randomization check
                    balance_checks = []
                    for var, values, name in [(family_income, 'Family Income'),
                                              (parental_education, 'Parental Education'),
                                              (baseline_achievement, 'Baseline Achievement')]:
                        treated_mean = np.mean(values[voucher == 1])
                        control_mean = np.mean(values[voucher == 0])
                        p_value = stats.ttest_ind(values[voucher == 1], values[voucher == 0])[1]

                        balance_checks.append({
                            'Variable': name,
                            'Treatment': f"{treated_mean:.0f}",
                            'Control': f"{control_mean:.0f}",
                            'p-value': f"{p_value:.3f}"
                        })

                    st.markdown(f"**Average Treatment Effect:** {ate_voucher:.1f} points")

                    st.markdown("**Randomization Check:**")
                    df_balance_check = pd.DataFrame(balance_checks)
                    st.dataframe(df_balance_check, use_container_width=True)

                with col2:
                    # Achievement distribution
                    fig = go.Figure()

                    fig.add_trace(go.Box(
                        y=post_achievement[voucher == 0],
                        name='No Voucher',
                        marker_color='red',
                        boxpoints='outliers'
                    ))

                    fig.add_trace(go.Box(
                        y=post_achievement[voucher == 1],
                        name='Voucher',
                        marker_color='blue',
                        boxpoints='outliers'
                    ))

                    fig.update_layout(
                        title="Achievement Distribution by Voucher Status",
                        yaxis_title="Test Score",
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Heterogeneous effects by income
                st.markdown("#### 💰 Effects by Family Income Level")

                income_effects = []
                income_labels = ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']

                for group in range(4):
                    mask = income_groups == group
                    if np.sum(mask & (voucher == 1)) > 10:
                        group_ate = (np.mean(post_achievement[mask & (voucher == 1)]) -
                                     np.mean(post_achievement[mask & (voucher == 0)]))

                        income_effects.append({
                            'Income Group': income_labels[group],
                            'Treatment Effect': group_ate,
                            'Sample Size': np.sum(mask),
                            'Mean Income': np.mean(family_income[mask])
                        })

                df_income_effects = pd.DataFrame(income_effects)

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(df_income_effects.round(1), use_container_width=True)

                with col2:
                    fig = go.Figure(data=[
                        go.Bar(x=df_income_effects['Income Group'],
                               y=df_income_effects['Treatment Effect'],
                               marker_color=['darkred', 'red', 'lightblue', 'blue'])
                    ])

                    fig.update_layout(
                        title="Voucher Effects by Income Level",
                        yaxis_title="Treatment Effect (Test Score Points)",
                        xaxis_title="Family Income Quartile"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown('<h3 class="subsection-header">💰 Economic Policy Applications</h3>', unsafe_allow_html=True)

            economic_example = st.selectbox(
                "Choose an economic application:",
                ["Minimum Wage Effects", "Tax Policy Analysis", "Social Safety Net Programs"]
            )

            if economic_example == "Minimum Wage Effects":
                st.markdown("""
                    <div class="example-box">
                    <h4>💼 Minimum Wage and Employment</h4>
                    <p>Using border discontinuity designs to estimate employment effects of minimum wage increases</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Minimum wage simulation using regression discontinuity design
                np.random.seed(101)
                n_counties = 200

                # County characteristics (running variable: distance from state border)
                distance_from_border = np.random.uniform(-100, 100, n_counties)  # Miles from border

                # Treatment: High minimum wage state (distance > 0)
                high_min_wage = (distance_from_border > 0).astype(int)

                # County characteristics
                population = np.random.lognormal(10, 1, n_counties)
                industry_mix = np.random.normal(0, 1, n_counties)  # Manufacturing intensity

                # Employment rate (outcome)
                # Smooth function of distance + discontinuous jump at border
                employment_rate = (75 + 0.05 * distance_from_border +
                                   0.1 * distance_from_border ** 2 / 1000 +
                                   2 * industry_mix +
                                   -3 * high_min_wage +  # True treatment effect
                                   np.random.normal(0, 2, n_counties))

                employment_rate = np.clip(employment_rate, 60, 90)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Regression Discontinuity Analysis")

                    # RD estimate using local linear regression around cutoff
                    bandwidth = 20  # Miles
                    local_sample = np.abs(distance_from_border) <= bandwidth

                    if np.sum(local_sample) > 10:
                        # Simple difference at cutoff
                        rd_estimate = (np.mean(employment_rate[local_sample & (distance_from_border > 0)]) -
                                       np.mean(employment_rate[local_sample & (distance_from_border <= 0)]))

                        st.markdown(f"""
                            **RD Estimates:**
                            - Bandwidth: ±{bandwidth} miles
                            - Treatment Effect: {rd_estimate:.2f} percentage points
                            - Local Sample Size: {np.sum(local_sample)} counties
                            """)

                    # Placebo test using predetermined variables
                    placebo_outcome = population  # Should not jump at cutoff
                    placebo_estimate = (np.mean(placebo_outcome[local_sample & (distance_from_border > 0)]) -
                                        np.mean(placebo_outcome[local_sample & (distance_from_border <= 0)]))

                    st.markdown(f"""
                        **Placebo Test (Population):**
                        - Jump at Cutoff: {placebo_estimate:.0f}
                        - Should be close to zero
                        """)

                with col2:
                    # RD plot
                    fig = go.Figure()

                    # Scatter plot
                    colors = ['red' if x <= 0 else 'blue' for x in distance_from_border]
                    fig.add_trace(go.Scatter(
                        x=distance_from_border,
                        y=employment_rate,
                        mode='markers',
                        marker=dict(color=colors, opacity=0.6),
                        name='Counties',
                        showlegend=False
                    ))

                    # Fitted lines on each side
                    left_side = distance_from_border <= 0
                    right_side = distance_from_border > 0

                    if np.sum(left_side) > 2:
                        left_fit = np.polyfit(distance_from_border[left_side], employment_rate[left_side], 1)
                        left_line_x = np.linspace(distance_from_border[left_side].min(), 0, 50)
                        left_line_y = np.polyval(left_fit, left_line_x)

                        fig.add_trace(go.Scatter(
                            x=left_line_x,
                            y=left_line_y,
                            mode='lines',
                            line=dict(color='red', width=3),
                            name='Low Min Wage State'
                        ))

                    if np.sum(right_side) > 2:
                        right_fit = np.polyfit(distance_from_border[right_side], employment_rate[right_side], 1)
                        right_line_x = np.linspace(0, distance_from_border[right_side].max(), 50)
                        right_line_y = np.polyval(right_fit, right_line_x)

                        fig.add_trace(go.Scatter(
                            x=right_line_x,
                            y=right_line_y,
                            mode='lines',
                            line=dict(color='blue', width=3),
                            name='High Min Wage State'
                        ))

                    # Vertical line at cutoff
                    fig.add_vline(x=0, line_dash="dash", line_color="black",
                                  annotation_text="State Border")

                    fig.update_layout(
                        title="Regression Discontinuity: Employment vs Distance",
                        xaxis_title="Distance from State Border (miles)",
                        yaxis_title="Employment Rate (%)"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Sensitivity analysis
                st.markdown("#### 🔍 Sensitivity Analysis")

                bandwidths = [10, 15, 20, 25, 30]
                sensitivity_results = []

                for bw in bandwidths:
                    local = np.abs(distance_from_border) <= bw
                    if np.sum(local & (distance_from_border > 0)) > 5 and np.sum(
                            local & (distance_from_border <= 0)) > 5:
                        estimate = (np.mean(employment_rate[local & (distance_from_border > 0)]) -
                                    np.mean(employment_rate[local & (distance_from_border <= 0)]))

                        sensitivity_results.append({
                            'Bandwidth': bw,
                            'Estimate': estimate,
                            'Sample Size': np.sum(local)
                        })

                df_sensitivity = pd.DataFrame(sensitivity_results)

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(df_sensitivity.round(3), use_container_width=True)

                with col2:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df_sensitivity['Bandwidth'],
                        y=df_sensitivity['Estimate'],
                        mode='lines+markers',
                        marker=dict(size=8),
                        line=dict(width=3)
                    ))

                    fig.add_hline(y=0, line_dash="dash", line_color="red")

                    fig.update_layout(
                        title="RD Estimates by Bandwidth",
                        xaxis_title="Bandwidth (miles)",
                        yaxis_title="Treatment Effect Estimate"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # Section 7: Practical Implementation
elif selected_section == "📊 Practical Implementation":
        st.markdown('<h2 class="section-header">📊 Practical Implementation</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🔧 Study Design", "📈 Power Analysis", "🎯 Estimation Methods"])

        with tab1:
            st.markdown('<h3 class="subsection-header">🔧 RCM-Based Study Design</h3>', unsafe_allow_html=True)

            st.markdown("""
                    <div class="important-box">
                    <h4>📋 Study Design Checklist</h4>
                    <p>Essential components for implementing RCM in practice</p>
                    </div>
                    """, unsafe_allow_html=True)

            design_phase = st.selectbox(
                "Select design phase:",
                ["Define Research Question", "Identify Treatment & Outcomes", "Address Confounding",
                 "Sample Size Planning"]
            )

            if design_phase == "Define Research Question":
                st.markdown("#### ❓ Formulating Causal Questions")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                            **Components of a Good Causal Question:**

                            1. **Population**: Who are we studying?
                            2. **Treatment**: What intervention are we comparing?
                            3. **Comparison**: What is the counterfactual?
                            4. **Outcome**: What are we measuring?
                            5. **Time Frame**: When do we measure effects?
                            """)

                    # Interactive example builder
                    st.markdown("#### 🛠️ Build Your Research Question")

                    population = st.text_input("Population:", "Adults aged 18-65 with diabetes")
                    treatment = st.text_input("Treatment:", "New medication")
                    comparison = st.text_input("Comparison:", "Standard care")
                    outcome = st.text_input("Outcome:", "HbA1c levels")
                    timeframe = st.text_input("Time Frame:", "6 months post-treatment")

                    if all([population, treatment, comparison, outcome, timeframe]):
                        research_question = f"""
                                **Your Research Question:**

                                Among {population}, what is the causal effect of {treatment} 
                                compared to {comparison} on {outcome} measured {timeframe}?
                                """
                        st.markdown(research_question)

                with col2:
                    st.markdown("#### ✅ Question Quality Assessment")

                    quality_criteria = [
                        ("Specific Population", "✅" if "population" in locals() and len(population) > 10 else "❌"),
                        ("Clear Treatment Definition", "✅" if "treatment" in locals() and len(treatment) > 5 else "❌"),
                        ("Appropriate Comparison", "✅" if "comparison" in locals() and len(comparison) > 5 else "❌"),
                        ("Measurable Outcome", "✅" if "outcome" in locals() and len(outcome) > 5 else "❌"),
                        ("Defined Time Frame", "✅" if "timeframe" in locals() and len(timeframe) > 5 else "❌")
                    ]

                    for criterion, status in quality_criteria:
                        st.markdown(f"- {criterion}: {status}")

                    # PICO framework visualization
                    pico_data = {
                        'Component': ['Population', 'Intervention', 'Comparison', 'Outcome'],
                        'Description': [
                            population if 'population' in locals() else 'Not specified',
                            treatment if 'treatment' in locals() else 'Not specified',
                            comparison if 'comparison' in locals() else 'Not specified',
                            outcome if 'outcome' in locals() else 'Not specified'
                        ]
                    }

                    st.markdown("**PICO Framework:**")
                    st.dataframe(pd.DataFrame(pico_data), use_container_width=True)

            elif design_phase == "Identify Treatment & Outcomes":
                st.markdown("#### 🎯 Treatment and Outcome Definition")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 💊 Treatment Definition")

                    treatment_type = st.selectbox(
                        "Treatment type:",
                        ["Binary (Yes/No)", "Multi-valued", "Continuous", "Time-varying"]
                    )

                    if treatment_type == "Binary (Yes/No)":
                        st.markdown("""
                                **Binary Treatment Examples:**
                                - Drug vs. Placebo
                                - Program participation vs. Non-participation
                                - Policy implementation vs. Status quo

                                **Key Considerations:**
                                - Clear definition of treatment receipt
                                - Intensity/dosage specification
                                - Compliance measurement
                                """)

                        # Treatment definition form
                        st.markdown("**Define Your Binary Treatment:**")
                        treat_name = st.text_input("Treatment name:")
                        treat_criteria = st.text_area("Treatment criteria:")
                        control_def = st.text_area("Control condition:")

                    elif treatment_type == "Multi-valued":
                        st.markdown("""
                                **Multi-valued Treatment Examples:**
                                - Different drug dosages (Low, Medium, High)
                                - Education levels (High school, College, Graduate)
                                - Training program types (Online, In-person, Hybrid)
                                """)

                    elif treatment_type == "Continuous":
                        st.markdown("""
                                **Continuous Treatment Examples:**
                                - Hours of training received
                                - Dosage amount (mg/day)
                                - Amount of financial aid
                                """)

                    else:  # Time-varying
                        st.markdown("""
                                **Time-varying Treatment Examples:**
                                - Treatment switching over time
                                - Varying intensity during study
                                - Dynamic treatment regimens
                                """)

                with col2:
                    st.markdown("#### 📊 Outcome Definition")

                    outcome_type = st.selectbox(
                        "Primary outcome type:",
                        ["Continuous", "Binary", "Count", "Time-to-event", "Composite"]
                    )

                    outcome_properties = {
                        "Continuous": {
                            "examples": ["Test scores", "Blood pressure", "Income"],
                            "considerations": ["Normality", "Outliers", "Measurement precision"]
                        },
                        "Binary": {
                            "examples": ["Graduation", "Employment", "Disease occurrence"],
                            "considerations": ["Event definition", "Time window", "Competing risks"]
                        },
                        "Count": {
                            "examples": ["Hospital visits", "Relapses", "Violations"],
                            "considerations": ["Over-dispersion", "Zero-inflation", "Exposure time"]
                        },
                        "Time-to-event": {
                            "examples": ["Survival time", "Time to graduation", "Employment duration"],
                            "considerations": ["Censoring", "Competing risks", "Time scale"]
                        },
                        "Composite": {
                            "examples": ["Multiple test scores", "Quality of life indices", "Safety endpoints"],
                            "considerations": ["Component weights", "Missing data", "Interpretation"]
                        }
                    }

                    props = outcome_properties[outcome_type]

                    st.markdown(f"**{outcome_type} Outcome Properties:**")
                    st.markdown("*Examples:*")
                    for example in props["examples"]:
                        st.markdown(f"- {example}")

                    st.markdown("*Key Considerations:*")
                    for consideration in props["considerations"]:
                        st.markdown(f"- {consideration}")

                    # Outcome measurement timeline
                    st.markdown("#### ⏰ Measurement Timeline")

                    timeline_points = st.multiselect(
                        "Select measurement points:",
                        ["Baseline", "Mid-treatment", "End of treatment", "1 month follow-up",
                         "3 months follow-up", "6 months follow-up", "1 year follow-up"],
                        default=["Baseline", "End of treatment"]
                    )

                    if timeline_points:
                        # Create timeline visualization
                        fig = go.Figure()

                        x_pos = list(range(len(timeline_points)))

                        fig.add_trace(go.Scatter(
                            x=x_pos,
                            y=[1] * len(timeline_points),
                            mode='markers+text',
                            marker=dict(size=15, color='blue'),
                            text=timeline_points,
                            textposition="top center",
                            showlegend=False
                        ))

                        # Connect with lines
                        fig.add_trace(go.Scatter(
                            x=x_pos,
                            y=[1] * len(timeline_points),
                            mode='lines',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        ))

                        fig.update_layout(
                            title="Measurement Timeline",
                            xaxis=dict(showticklabels=False),
                            yaxis=dict(showticklabels=False, range=[0.8, 1.4]),
                            height=200
                        )

                        st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown('<h3 class="subsection-header">📈 Power Analysis for RCM Studies</h3>', unsafe_allow_html=True)

            st.markdown("""
                    <div class="important-box">
                    <h4>⚡ Statistical Power in Causal Inference</h4>
                    <p>Power analysis helps determine the minimum sample size needed to detect meaningful treatment effects</p>
                    </div>
                    """, unsafe_allow_html=True)

            power_type = st.selectbox(
                "Select power analysis type:",
                ["Two-sample t-test", "Randomized Trial", "Observational Study with Confounders"]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ⚙️ Power Analysis Parameters")

                # Common parameters
                alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01)
                power = st.slider("Desired power (1-β):", 0.70, 0.99, 0.80, 0.01)

                if power_type == "Two-sample t-test":
                    effect_size = st.slider("Effect size (Cohen's d):", 0.1, 2.0, 0.5, 0.1)
                    allocation_ratio = st.slider("Treatment:Control ratio:", 0.1, 2.0, 1.0, 0.1)

                    # Calculate sample size
                    from scipy import stats

                    # Two-sided test
                    z_alpha = stats.norm.ppf(1 - alpha / 2)
                    z_beta = stats.norm.ppf(power)

                    # Sample size formula for two-sample t-test
                    n_per_group = ((z_alpha + z_beta) ** 2 * 2 * (1 + 1 / allocation_ratio)) / (effect_size ** 2)
                    n_treatment = int(np.ceil(n_per_group))
                    n_control = int(np.ceil(n_per_group * allocation_ratio))
                    n_total = n_treatment + n_control

                    st.markdown(f"""
                            **Sample Size Calculation:**
                            - Treatment group: {n_treatment}
                            - Control group: {n_control}
                            - Total sample size: {n_total}
                            """)

                elif power_type == "Randomized Trial":
                    baseline_mean = st.number_input("Baseline outcome mean:", value=100.0)
                    baseline_sd = st.number_input("Baseline outcome SD:", value=15.0)
                    treatment_effect = st.number_input("Expected treatment effect:", value=10.0)
                    dropout_rate = st.slider("Expected dropout rate:", 0.0, 0.5, 0.1, 0.05)

                    # Effect size
                    effect_size = treatment_effect / baseline_sd

                    # Sample size calculation
                    z_alpha = stats.norm.ppf(1 - alpha / 2)
                    z_beta = stats.norm.ppf(power)

                    n_per_group = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)

                    # Adjust for dropout
                    n_adjusted = n_per_group / (1 - dropout_rate)

                    n_total_adjusted = int(np.ceil(2 * n_adjusted))

                    st.markdown(f"""
                            **Sample Size Calculation:**
                            - Effect size (Cohen's d): {effect_size:.2f}
                            - Per group (before dropout): {int(np.ceil(n_per_group))}
                            - Per group (after dropout): {int(np.ceil(n_adjusted))}
                            - Total sample size: {n_total_adjusted}
                            """)

            with col2:
                st.markdown("#### 📊 Power Curve Analysis")

                # Generate power curves
                if power_type in ["Two-sample t-test", "Randomized Trial"]:
                    sample_sizes = np.arange(10, 500, 10)

                    if power_type == "Two-sample t-test":
                        effect_sizes_to_plot = [0.2, 0.5, 0.8]  # Small, medium, large
                    else:
                        effect_sizes_to_plot = [treatment_effect / baseline_sd * 0.5,
                                                treatment_effect / baseline_sd,
                                                treatment_effect / baseline_sd * 1.5]

                    fig = go.Figure()

                    colors = ['red', 'blue', 'green']

                    for i, es in enumerate(effect_sizes_to_plot):
                        powers = []

                        for n in sample_sizes:
                            # Power calculation
                            z_alpha = stats.norm.ppf(1 - alpha / 2)
                            delta = es * np.sqrt(n / 2)  # Non-centrality parameter
                            power_calc = 1 - stats.norm.cdf(z_alpha - delta) + stats.norm.cdf(-z_alpha - delta)
                            powers.append(power_calc)

                        fig.add_trace(go.Scatter(
                            x=sample_sizes,
                            y=powers,
                            mode='lines',
                            name=f'Effect size = {es:.2f}',
                            line=dict(color=colors[i], width=3)
                        ))

                    fig.add_hline(y=0.8, line_dash="dash", line_color="black",
                                  annotation_text="80% Power")

                    fig.update_layout(
                        title="Statistical Power vs Sample Size",
                        xaxis_title="Total Sample Size",
                        yaxis_title="Statistical Power",
                        yaxis=dict(range=[0, 1])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Minimum detectable effect size
                st.markdown("#### 🎯 Minimum Detectable Effect Size")

                fixed_n = st.number_input("Fixed sample size:", value=200, min_value=10)

                # Calculate MDES
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(power)

                mdes = (z_alpha + z_beta) * np.sqrt(2 / fixed_n)

                st.markdown(f"""
                        **With n={fixed_n}:**
                        - Minimum detectable effect size: {mdes:.3f}
                        - In original units: {mdes * baseline_sd if 'baseline_sd' in locals() else 'N/A'}
                        """)

        with tab3:
            st.markdown('<h3 class="subsection-header">🎯 Causal Effect Estimation Methods</h3>', unsafe_allow_html=True)

            estimation_method = st.selectbox(
                "Select estimation method:",
                ["Difference in Means", "ANCOVA", "Propensity Score Methods", "Instrumental Variables",
                 "Regression Discontinuity"]
            )

            if estimation_method == "Difference in Means":
                st.markdown("""
                        <div class="important-box">
                        <h4>📊 Simple Difference in Means</h4>
                        <p>The most basic estimator for randomized experiments</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Simulate simple RCT data
                np.random.seed(42)
                n_sim = st.slider("Sample size:", 50, 500, 200)
                true_effect_sim = st.slider("True treatment effect:", -10, 10, 5)
                noise_level = st.slider("Noise level (SD):", 1, 20, 10)

                # Generate data
                treatment_sim = np.random.binomial(1, 0.5, n_sim)
                outcome_sim = 50 + true_effect_sim * treatment_sim + np.random.normal(0, noise_level, n_sim)

                # Estimation
                y1_mean = np.mean(outcome_sim[treatment_sim == 1])
                y0_mean = np.mean(outcome_sim[treatment_sim == 0])
                ate_estimate = y1_mean - y0_mean

                # Standard error
                n1 = np.sum(treatment_sim == 1)
                n0 = np.sum(treatment_sim == 0)
                s1_squared = np.var(outcome_sim[treatment_sim == 1], ddof=1)
                s0_squared = np.var(outcome_sim[treatment_sim == 0], ddof=1)
                se_ate = np.sqrt(s1_squared / n1 + s0_squared / n0)

                # Confidence interval
                ci_lower = ate_estimate - 1.96 * se_ate
                ci_upper = ate_estimate + 1.96 * se_ate

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Estimation Results")

                    st.markdown(f"""
                            **ATE Estimation:**
                            - Treatment mean: {y1_mean:.2f}
                            - Control mean: {y0_mean:.2f}
                            - **ATE estimate: {ate_estimate:.2f}**
                            - Standard error: {se_ate:.2f}
                            - 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]
                            - True effect: {true_effect_sim}
                            """)

                    # T-test
                    t_stat = ate_estimate / se_ate
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_sim - 2))

                    st.markdown(f"""
                            **Statistical Test:**
                            - t-statistic: {t_stat:.3f}
                            - p-value: {p_value:.4f}
                            - Significant: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)
                            """)

                with col2:
                    # Distribution comparison
                    fig = go.Figure()

                    fig.add_trace(go.Box(
                        y=outcome_sim[treatment_sim == 0],
                        name='Control',
                        marker_color='red',
                        boxpoints='outliers'
                    ))

                    fig.add_trace(go.Box(
                        y=outcome_sim[treatment_sim == 1],
                        name='Treatment',
                        marker_color='blue',
                        boxpoints='outliers'
                    ))

                    fig.update_layout(
                        title="Outcome Distributions by Treatment",
                        yaxis_title="Outcome",
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif estimation_method == "ANCOVA":
                st.markdown("""
                        <div class="important-box">
                        <h4>📈 ANCOVA (Analysis of Covariance)</h4>
                        <p>Adjusting for baseline covariates to improve precision</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Mathematical formula
                st.latex(r'''
                        Y_i = \alpha + \tau D_i + \beta X_i + \epsilon_i
                        ''')

                st.markdown("Where:")
                st.markdown("- $Y_i$ = outcome for unit i")
                st.markdown("- $D_i$ = treatment indicator")
                st.markdown("- $X_i$ = baseline covariate")
                st.markdown("- $\\tau$ = treatment effect (our parameter of interest)")

                # Simulate ANCOVA data
                np.random.seed(123)
                n_ancova = 200

                # Generate baseline covariate
                X = np.random.normal(0, 1, n_ancova)

                # Treatment assignment
                D = np.random.binomial(1, 0.5, n_ancova)

                # Outcome with covariate relationship
                beta_true = 2  # Covariate effect
                tau_true = 3  # Treatment effect
                Y = 10 + tau_true * D + beta_true * X + np.random.normal(0, 2, n_ancova)

                # Compare estimators
                from sklearn.linear_model import LinearRegression

                # Simple difference (ignoring X)
                simple_ate = np.mean(Y[D == 1]) - np.mean(Y[D == 0])

                # ANCOVA estimator
                reg = LinearRegression()
                reg.fit(np.column_stack([D, X]), Y)
                ancova_ate = reg.coef_[0]

                # Standard errors (simplified)
                residuals = Y - reg.predict(np.column_stack([D, X]))
                mse = np.mean(residuals ** 2)

                # Design matrix
                Z = np.column_stack([np.ones(n_ancova), D, X])
                se_ancova = np.sqrt(mse * np.linalg.inv(Z.T @ Z)[1, 1])

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Estimator Comparison")

                    st.markdown(f"""
                            **Simple Difference in Means:**
                            - Estimate: {simple_ate:.3f}
                            - True effect: {tau_true}
                            - Bias: {simple_ate - tau_true:.3f}

                            **ANCOVA Estimator:**
                            - Estimate: {ancova_ate:.3f}
                            - Standard error: {se_ancova:.3f}
                            - 95% CI: [{ancova_ate - 1.96 * se_ancova:.3f}, {ancova_ate + 1.96 * se_ancova:.3f}]
                            - Bias: {ancova_ate - tau_true:.3f}
                            """)

                    # Efficiency gain
                    simple_var = np.var(Y[D == 1]) / np.sum(D == 1) + np.var(Y[D == 0]) / np.sum(D == 0)
                    efficiency_gain = simple_var / se_ancova ** 2

                    st.markdown(f"""
                            **Efficiency Gain:**
                            - Variance ratio: {efficiency_gain:.2f}
                            - ANCOVA is {efficiency_gain:.1f}x more efficient
                            """)

                with col2:
                    # Residual plots
                    fig = make_subplots(rows=2, cols=1,
                                        subplot_titles=["Before Adjustment", "After Adjustment"])

                    # Before adjustment
                    fig.add_trace(go.Scatter(
                        x=X[D == 0], y=Y[D == 0],
                        mode='markers', name='Control',
                        marker=dict(color='red', opacity=0.6)
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=X[D == 1], y=Y[D == 1],
                        mode='markers', name='Treatment',
                        marker=dict(color='blue', opacity=0.6)
                    ), row=1, col=1)

                    # After adjustment (residuals)
                    Y_adjusted = Y - beta_true * X

                    fig.add_trace(go.Scatter(
                        x=X[D == 0], y=Y_adjusted[D == 0],
                        mode='markers', name='Control (Adjusted)',
                        marker=dict(color='red', opacity=0.6), showlegend=False
                    ), row=2, col=1)

                    fig.add_trace(go.Scatter(
                        x=X[D == 1], y=Y_adjusted[D == 1],
                        mode='markers', name='Treatment (Adjusted)',
                        marker=dict(color='blue', opacity=0.6), showlegend=False
                    ), row=2, col=1)

                    fig.update_layout(
                        title="ANCOVA: Before and After Covariate Adjustment",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # Section 8: Result Interpretation
elif selected_section == "🔍 Result Interpretation":
    st.markdown('<h2 class="section-header">🔍 Interpreting RCM Results</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Effect Sizes", "📈 Statistical Significance", "🎯 Clinical/Practical Significance",
         "⚠️ Limitations & Caveats"])

    with tab1:
        st.markdown('<h3 class="subsection-header">📊 Understanding Effect Sizes</h3>', unsafe_allow_html=True)

        st.markdown("""
                    <div class="important-box">
                    <h4>📏 Effect Size Interpretation</h4>
                    <p>Effect sizes quantify the magnitude of treatment effects in standardized units</p>
                    </div>
                    """, unsafe_allow_html=True)

        effect_type = st.selectbox(
            "Select effect size type:",
            ["Cohen's d", "Odds Ratio", "Risk Difference", "Number Needed to Treat"]
        )

        if effect_type == "Cohen's d":
            st.markdown("#### 📐 Cohen's d for Continuous Outcomes")

            st.latex(r'''
                        d = \frac{\bar{Y}_1 - \bar{Y}_0}{s_{pooled}}
                        ''')

            st.markdown("Where $s_{pooled} = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_0-1)s_0^2}{n_1 + n_0 - 2}}$")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎯 Interactive Calculator")

                mean_treatment = st.number_input("Treatment group mean:", value=55.0)
                mean_control = st.number_input("Control group mean:", value=50.0)
                sd_treatment = st.number_input("Treatment group SD:", value=10.0)
                sd_control = st.number_input("Control group SD:", value=10.0)
                n_treatment = st.number_input("Treatment group size:", value=100, min_value=2)
                n_control = st.number_input("Control group size:", value=100, min_value=2)

                # Calculate Cohen's d
                pooled_sd = np.sqrt(((n_treatment - 1) * sd_treatment ** 2 + (n_control - 1) * sd_control ** 2) /
                                    (n_treatment + n_control - 2))
                cohens_d = (mean_treatment - mean_control) / pooled_sd

                # Interpretation
                if abs(cohens_d) < 0.2:
                    interpretation = "Negligible"
                    color = "gray"
                elif abs(cohens_d) < 0.5:
                    interpretation = "Small"
                    color = "orange"
                elif abs(cohens_d) < 0.8:
                    interpretation = "Medium"
                    color = "blue"
                else:
                    interpretation = "Large"
                    color = "green"

                st.markdown(f"""
                            **Results:**
                            - Pooled SD: {pooled_sd:.2f}
                            - **Cohen's d: {cohens_d:.3f}**
                            - Interpretation: <span style="color: {color}; font-weight: bold;">{interpretation}</span>
                            """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 📊 Effect Size Visualization")

                # Create overlapping distributions
                x = np.linspace(mean_control - 4 * sd_control, mean_treatment + 4 * sd_treatment, 1000)

                # Control distribution
                y_control = stats.norm.pdf(x, mean_control, sd_control)
                # Treatment distribution
                y_treatment = stats.norm.pdf(x, mean_treatment, sd_treatment)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=x, y=y_control,
                    mode='lines',
                    name='Control',
                    line=dict(color='red', width=3),
                    fill='tonexty'
                ))

                fig.add_trace(go.Scatter(
                    x=x, y=y_treatment,
                    mode='lines',
                    name='Treatment',
                    line=dict(color='blue', width=3),
                    fill='tonexty'
                ))

                # Add vertical lines for means
                fig.add_vline(x=mean_control, line_dash="dash", line_color="red",
                              annotation_text="Control Mean")
                fig.add_vline(x=mean_treatment, line_dash="dash", line_color="blue",
                              annotation_text="Treatment Mean")

                fig.update_layout(
                    title=f"Distribution Overlap (Cohen's d = {cohens_d:.2f})",
                    xaxis_title="Outcome Value",
                    yaxis_title="Density"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Effect size benchmarks
                st.markdown("#### 📏 Cohen's d Benchmarks")
                benchmarks = pd.DataFrame({
                    'Effect Size': ['0.2', '0.5', '0.8'],
                    'Interpretation': ['Small', 'Medium', 'Large'],
                    'Overlap': ['85%', '67%', '53%']
                })
                st.dataframe(benchmarks, use_container_width=True)

        elif effect_type == "Odds Ratio":
            st.markdown("#### 🎲 Odds Ratio for Binary Outcomes")

            st.latex(r'''
                        OR = \frac{P(Y=1|D=1)/(1-P(Y=1|D=1))}{P(Y=1|D=0)/(1-P(Y=1|D=0))}
                        ''')

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 2x2 Contingency Table")

                # Interactive 2x2 table
                a = st.number_input("Treatment Success (a):", value=40, min_value=0)
                b = st.number_input("Treatment Failure (b):", value=60, min_value=0)
                c = st.number_input("Control Success (c):", value=20, min_value=0)
                d = st.number_input("Control Failure (d):", value=80, min_value=0)

                # Create contingency table
                contingency_data = {
                    '': ['Treatment', 'Control', 'Total'],
                    'Success': [a, c, a + c],
                    'Failure': [b, d, b + d],
                    'Total': [a + b, c + d, a + b + c + d]
                }

                st.dataframe(pd.DataFrame(contingency_data), use_container_width=True)

                # Calculate measures
                if b > 0 and d > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)

                    # Risk measures
                    risk_treatment = a / (a + b)
                    risk_control = c / (c + d)
                    risk_difference = risk_treatment - risk_control
                    relative_risk = risk_treatment / risk_control if risk_control > 0 else np.inf

                    st.markdown(f"""
                                **Calculated Measures:**
                                - **Odds Ratio: {odds_ratio:.3f}**
                                - Risk Difference: {risk_difference:.3f}
                                - Relative Risk: {relative_risk:.3f}
                                - Treatment Risk: {risk_treatment:.3f}
                                - Control Risk: {risk_control:.3f}
                                """)
                else:
                    st.warning("Please ensure all cells have positive values for valid calculations.")

            with col2:
                if 'odds_ratio' in locals():
                    # OR interpretation
                    if odds_ratio > 1:
                        or_interpretation = f"Treatment increases odds by {(odds_ratio - 1) * 100:.1f}%"
                        or_color = "green" if odds_ratio > 1.5 else "orange"
                    elif odds_ratio < 1:
                        or_interpretation = f"Treatment decreases odds by {(1 - odds_ratio) * 100:.1f}%"
                        or_color = "red"
                    else:
                        or_interpretation = "No effect"
                        or_color = "gray"

                    st.markdown(f"""
                                #### 📈 Interpretation

                                **Odds Ratio = {odds_ratio:.3f}**

                                <span style="color: {or_color}; font-weight: bold;">{or_interpretation}</span>
                                """, unsafe_allow_html=True)

                    # Confidence interval (approximate)
                    log_or = np.log(odds_ratio)
                    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                    ci_lower = np.exp(log_or - 1.96 * se_log_or)
                    ci_upper = np.exp(log_or + 1.96 * se_log_or)

                    st.markdown(f"""
                                **95% Confidence Interval:**
                                [{ci_lower:.3f}, {ci_upper:.3f}]
                                """)

                    # Visual representation
                    fig = go.Figure()

                    categories = ['Treatment', 'Control']
                    success_rates = [risk_treatment, risk_control]
                    failure_rates = [1 - risk_treatment, 1 - risk_control]

                    fig.add_trace(go.Bar(
                        x=categories,
                        y=success_rates,
                        name='Success',
                        marker_color='green'
                    ))

                    fig.add_trace(go.Bar(
                        x=categories,
                        y=failure_rates,
                        name='Failure',
                        marker_color='red'
                    ))

                    fig.update_layout(
                        title="Success/Failure Rates by Group",
                        yaxis_title="Proportion",
                        barmode='stack'
                    )

                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<h3 class="subsection-header">📈 Statistical Significance Testing</h3>', unsafe_allow_html=True)

        st.markdown("""
                    <div class="warning-box">
                    <h4>⚠️ P-values vs Effect Sizes</h4>
                    <p>Statistical significance (p < 0.05) does not imply practical importance. Always consider effect sizes alongside p-values.</p>
                    </div>
                    """, unsafe_allow_html=True)

        sig_test_type = st.selectbox(
            "Select significance test:",
            ["t-test for Means", "Chi-square Test", "Multiple Testing Correction"]
        )

        if sig_test_type == "t-test for Means":
            st.markdown("#### 🧮 Two-Sample t-test")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Test Parameters")

                # Generate or input data
                data_source = st.radio("Data source:", ["Simulate data", "Enter summary statistics"])

                if data_source == "Simulate data":
                    n1_t = st.slider("Treatment group size:", 10, 200, 50)
                    n0_t = st.slider("Control group size:", 10, 200, 50)
                    mean_diff = st.slider("True mean difference:", -10, 10, 3)
                    common_sd = st.slider("Common standard deviation:", 1, 20, 8)

                    # Generate data
                    np.random.seed(42)
                    data_treatment = np.random.normal(50 + mean_diff, common_sd, n1_t)
                    data_control = np.random.normal(50, common_sd, n0_t)

                    # Calculate statistics
                    mean1 = np.mean(data_treatment)
                    mean0 = np.mean(data_control)
                    s1 = np.std(data_treatment, ddof=1)
                    s0 = np.std(data_control, ddof=1)

                else:
                    mean1 = st.number_input("Treatment mean:", value=53.0)
                    mean0 = st.number_input("Control mean:", value=50.0)
                    s1 = st.number_input("Treatment SD:", value=8.0)
                    s0 = st.number_input("Control SD:", value=8.0)
                    n1_t = st.number_input("Treatment n:", value=50, min_value=2)
                    n0_t = st.number_input("Control n:", value=50, min_value=2)

                # Perform t-test
                observed_diff = mean1 - mean0
                pooled_se = np.sqrt(s1 ** 2 / n1_t + s0 ** 2 / n0_t)
                t_statistic = observed_diff / pooled_se

                # Degrees of freedom (Welch's t-test)
                df = (s1 ** 2 / n1_t + s0 ** 2 / n0_t) ** 2 / (
                            (s1 ** 2 / n1_t) ** 2 / (n1_t - 1) + (s0 ** 2 / n0_t) ** 2 / (n0_t - 1))

                # P-value (two-tailed)
                p_value_two = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

                st.markdown(f"""
                            **Test Results:**
                            - Observed difference: {observed_diff:.3f}
                            - Standard error: {pooled_se:.3f}
                            - t-statistic: {t_statistic:.3f}
                            - Degrees of freedom: {df:.1f}
                            - **p-value: {p_value_two:.6f}**
                            - Significant: {'Yes' if p_value_two < 0.05 else 'No'} (α = 0.05)
                            """)

            with col2:
                st.markdown("#### 📊 t-Distribution Visualization")

                # Plot t-distribution with test statistic
                x_t = np.linspace(-4, 4, 1000)
                y_t = stats.t.pdf(x_t, df)

                fig = go.Figure()

                # Plot distribution
                fig.add_trace(go.Scatter(
                    x=x_t, y=y_t,
                    mode='lines',
                    name='t-distribution',
                    line=dict(color='blue', width=2)
                ))

                # Shade critical regions
                alpha_level = 0.05
                t_critical = stats.t.ppf(1 - alpha_level / 2, df)

                # Right tail
                x_right = x_t[x_t >= t_critical]
                y_right = stats.t.pdf(x_right, df)
                fig.add_trace(go.Scatter(
                    x=x_right, y=y_right,
                    fill='tonexty',
                    mode='lines',
                    name='Critical region',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ))

                # Left tail
                x_left = x_t[x_t <= -t_critical]
                y_left = stats.t.pdf(x_left, df)
                fig.add_trace(go.Scatter(
                    x=x_left, y=y_left,
                    fill='tonexty',
                    mode='lines',
                    name='Critical region',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)',
                    showlegend=False
                ))

                # Mark observed t-statistic
                fig.add_vline(x=t_statistic, line_dash="dash", line_color="green",
                              annotation_text=f"t = {t_statistic:.2f}")
                fig.add_vline(x=-t_statistic, line_dash="dash", line_color="green")

                fig.update_layout(
                    title=f"t-Distribution (df = {df:.1f})",
                    xaxis_title="t-value",
                    yaxis_title="Density"
                )

                st.plotly_chart(fig, use_container_width=True)

                # P-value interpretation
                if p_value_two < 0.001:
                    p_interpretation = "Very strong evidence against null hypothesis"
                elif p_value_two < 0.01:
                    p_interpretation = "Strong evidence against null hypothesis"
                elif p_value_two < 0.05:
                    p_interpretation = "Moderate evidence against null hypothesis"
                elif p_value_two < 0.1:
                    p_interpretation = "Weak evidence against null hypothesis"
                else:
                    p_interpretation = "Little to no evidence against null hypothesis"

                st.markdown(f"""
                            **P-value Interpretation:**
                            {p_interpretation}
                            """)

    with tab3:
        st.markdown('<h3 class="subsection-header">🎯 Clinical and Practical Significance</h3>', unsafe_allow_html=True)

        st.markdown("""
                    <div class="important-box">
                    <h4>🏥 Beyond Statistical Significance</h4>
                    <p>Clinical/practical significance considers whether the effect size is meaningful in real-world contexts</p>
                    </div>
                    """, unsafe_allow_html=True)

        context_type = st.selectbox(
            "Select application context:",
            ["Medical Treatment", "Educational Intervention", "Economic Policy", "Psychological Therapy"]
        )

        if context_type == "Medical Treatment":
            st.markdown("#### 🏥 Medical Treatment Evaluation")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Treatment Results")

                # Input clinical data
                outcome_type_med = st.selectbox(
                    "Outcome type:",
                    ["Blood Pressure Reduction", "Cholesterol Change", "Symptom Scale", "Survival Months"]
                )

                if outcome_type_med == "Blood Pressure Reduction":
                    treatment_effect = st.slider("Treatment effect (mmHg):", -30, 0, -10)
                    baseline_value = st.slider("Baseline BP (mmHg):", 120, 200, 160)

                    # Clinical significance thresholds
                    minimal_clinically_important = -5  # 5 mmHg reduction
                    substantial_benefit = -10  # 10 mmHg reduction

                    st.markdown(f"""
                                **Clinical Significance Assessment:**
                                - Treatment effect: {treatment_effect} mmHg
                                - Baseline BP: {baseline_value} mmHg
                                - Post-treatment BP: {baseline_value + treatment_effect} mmHg

                                **Clinical Thresholds:**
                                - Minimal important difference: {minimal_clinically_important} mmHg
                                - Substantial benefit: {substantial_benefit} mmHg
                                """)

                    # Determine clinical significance
                    if treatment_effect <= substantial_benefit:
                        clinical_sig = "Clinically substantial"
                        color = "green"
                    elif treatment_effect <= minimal_clinically_important:
                        clinical_sig = "Clinically meaningful"
                        color = "orange"
                    else:
                        clinical_sig = "Clinically insignificant"
                        color = "red"

                    st.markdown(f"""
                                **Assessment: <span style="color: {color}; font-weight: bold;">{clinical_sig}</span>**
                                """, unsafe_allow_html=True)

                elif outcome_type_med == "Survival Months":
                    survival_benefit = st.slider("Survival benefit (months):", 0, 24, 6)
                    baseline_survival = st.slider("Baseline median survival (months):", 6, 60, 18)

                    # Calculate relative improvement
                    relative_improvement = survival_benefit / baseline_survival * 100

                    st.markdown(f"""
                                **Survival Analysis:**
                                - Survival benefit: {survival_benefit} months
                                - Baseline survival: {baseline_survival} months
                                - Relative improvement: {relative_improvement:.1f}%
                                """)

                    # Clinical significance for survival
                    if survival_benefit >= 6:
                        survival_sig = "Clinically meaningful"
                        color = "green"
                    elif survival_benefit >= 3:
                        survival_sig = "Potentially meaningful"
                        color = "orange"
                    else:
                        survival_sig = "Clinically questionable"
                        color = "red"

                    st.markdown(f"""
                                **Assessment: <span style="color: {color}; font-weight: bold;">{survival_sig}</span>**
                                """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 💰 Cost-Effectiveness Considerations")

                # Cost-effectiveness analysis
                treatment_cost = st.number_input("Treatment cost per patient ($):", value=5000, min_value=0)
                control_cost = st.number_input("Control/standard care cost ($):", value=2000, min_value=0)

                if outcome_type_med == "Blood Pressure Reduction":
                    # Quality-adjusted life years (simplified)
                    bp_reduction_qaly = abs(treatment_effect) * 0.01  # Rough approximation
                    cost_per_qaly = (
                                                treatment_cost - control_cost) / bp_reduction_qaly if bp_reduction_qaly > 0 else np.inf

                    st.markdown(f"""
                                **Cost-Effectiveness:**
                                - Additional cost: ${treatment_cost - control_cost:,.0f}
                                - Estimated QALY gain: {bp_reduction_qaly:.3f}
                                - Cost per QALY: ${cost_per_qaly:,.0f}
                                """)

                    # Cost-effectiveness thresholds
                    if cost_per_qaly <= 50000:
                        ce_assessment = "Highly cost-effective"
                        ce_color = "green"
                    elif cost_per_qaly <= 100000:
                        ce_assessment = "Cost-effective"
                        ce_color = "orange"
                    else:
                        ce_assessment = "Not cost-effective"
                        ce_color = "red"

                    st.markdown(f"""
                                **Assessment: <span style="color: {ce_color}; font-weight: bold;">{ce_assessment}</span>**

                                *Common threshold: $50,000-$100,000 per QALY*
                                """, unsafe_allow_html=True)

                elif outcome_type_med == "Survival Months":
                    cost_per_life_month = (
                                                      treatment_cost - control_cost) / survival_benefit if survival_benefit > 0 else np.inf
                    cost_per_life_year = cost_per_life_month * 12

                    st.markdown(f"""
                                **Cost-Effectiveness:**
                                - Additional cost: ${treatment_cost - control_cost:,.0f}
                                - Cost per life-month: ${cost_per_life_month:,.0f}
                                - Cost per life-year: ${cost_per_life_year:,.0f}
                                """)

                # Number needed to treat (for binary outcomes)
                st.markdown("#### 📊 Number Needed to Treat (NNT)")

                control_event_rate = st.slider("Control group event rate:", 0.0, 1.0, 0.3, 0.01)
                treatment_event_rate = st.slider("Treatment group event rate:", 0.0, 1.0, 0.2, 0.01)

                if abs(control_event_rate - treatment_event_rate) > 0.001:
                    nnt = 1 / abs(control_event_rate - treatment_event_rate)

                    st.markdown(f"""
                                **NNT Analysis:**
                                - Control event rate: {control_event_rate:.1%}
                                - Treatment event rate: {treatment_event_rate:.1%}
                                - Absolute risk reduction: {abs(control_event_rate - treatment_event_rate):.1%}
                                - **Number needed to treat: {nnt:.1f}**
                                """)

                    if nnt <= 10:
                        nnt_interpretation = "Very effective"
                    elif nnt <= 25:
                        nnt_interpretation = "Moderately effective"
                    elif nnt <= 100:
                        nnt_interpretation = "Marginally effective"
                    else:
                        nnt_interpretation = "Limited effectiveness"

                    st.markdown(f"*Interpretation: {nnt_interpretation}*")
                else:
                    st.markdown("No difference in event rates")

    with tab4:
        st.markdown('<h3 class="subsection-header">⚠️ Limitations and Caveats</h3>', unsafe_allow_html=True)

        st.markdown("""
                    <div class="warning-box">
                    <h4>🚨 Important Limitations of RCM</h4>
                    <p>Understanding the boundaries and assumptions of causal inference</p>
                    </div>
                    """, unsafe_allow_html=True)

        limitation_category = st.selectbox(
            "Select limitation category:",
            ["Fundamental Assumptions", "External Validity", "Interference & Spillovers", "Missing Data Issues"]
        )

        if limitation_category == "Fundamental Assumptions":
            st.markdown("#### 🔍 SUTVA and Other Key Assumptions")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                            **Stable Unit Treatment Value Assumption (SUTVA):**

                            1. **No interference between units**
                               - Treatment of one unit doesn't affect others
                               - Violations: network effects, spillovers, general equilibrium effects

                            2. **Consistency/well-defined treatments**
                               - Same treatment value implies same potential outcome
                               - Violations: treatment heterogeneity, implementation differences
                            """)

                # Interactive SUTVA violation examples
                sutva_example = st.selectbox(
                    "Choose SUTVA violation example:",
                    ["Vaccination spillovers", "School peer effects", "Market equilibrium effects"]
                )

                if sutva_example == "Vaccination spillovers":
                    st.markdown("""
                                <div class="example-box">
                                <h5>💉 Vaccination Program Example</h5>
                                <p><strong>Violation:</strong> Vaccinating some individuals protects unvaccinated neighbors (herd immunity)</p>
                                <p><strong>Consequence:</strong> Underestimates true program benefits</p>
                                <p><strong>Solutions:</strong> Cluster randomization, network analysis methods</p>
                                </div>
                                """, unsafe_allow_html=True)

                elif sutva_example == "School peer effects":
                    st.markdown("""
                                <div class="example-box">
                                <h5>🏫 Educational Intervention Example</h5>
                                <p><strong>Violation:</strong> Treated students affect learning of non-treated classmates</p>
                                <p><strong>Consequence:</strong> Biased estimates of individual treatment effects</p>
                                <p><strong>Solutions:</strong> School-level randomization, peer effect models</p>
                                </div>
                                """, unsafe_allow_html=True)

                elif sutva_example == "Market equilibrium effects":
                    st.markdown("""
                                <div class="example-box">
                                <h5>💼 Job Training Program Example</h5>
                                <p><strong>Violation:</strong> Training some workers affects wages/employment of others</p>
                                <p><strong>Consequence:</strong> Partial vs. general equilibrium effects differ</p>
                                <p><strong>Solutions:</strong> Large-scale experiments, equilibrium models</p>
                                </div>
                                """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🎯 Assumption Violations Simulation")

                # Simulate SUTVA violation
                np.random.seed(42)
                n_units = 100

                # Network structure (simplified)
                network_strength = st.slider("Network/spillover strength:", 0.0, 1.0, 0.3, 0.1)

                # Generate treatment assignment
                treatment_prob = 0.5
                treatment = np.random.binomial(1, treatment_prob, n_units)

                # True individual treatment effect
                individual_effect = 2.0

                # Spillover effect (proportion of treated neighbors)
                spillover_effect = 1.0
                neighbors_treated = np.random.binomial(3, treatment_prob, n_units) / 3  # 3 neighbors on average

                # Outcomes with spillovers
                baseline_outcome = np.random.normal(10, 2, n_units)

                # Without spillovers (SUTVA holds)
                outcome_no_spillover = baseline_outcome + individual_effect * treatment + np.random.normal(0, 1,
                                                                                                           n_units)

                # With spillovers (SUTVA violated)
                outcome_with_spillover = (baseline_outcome +
                                          individual_effect * treatment +
                                          network_strength * spillover_effect * neighbors_treated +
                                          np.random.normal(0, 1, n_units))

                # Estimate treatment effects
                ate_no_spillover = np.mean(outcome_no_spillover[treatment == 1]) - np.mean(
                    outcome_no_spillover[treatment == 0])
                ate_with_spillover = np.mean(outcome_with_spillover[treatment == 1]) - np.mean(
                    outcome_with_spillover[treatment == 0])

                # True total effect (direct + spillover)
                true_total_effect = individual_effect + network_strength * spillover_effect * treatment_prob

                st.markdown(f"""
                            **SUTVA Violation Demonstration:**

                            - True individual effect: {individual_effect:.1f}
                            - True total effect: {true_total_effect:.1f}
                            - Estimated ATE (no spillover): {ate_no_spillover:.2f}
                            - Estimated ATE (with spillover): {ate_with_spillover:.2f}
                            - Bias from ignoring spillovers: {ate_no_spillover - true_total_effect:.2f}
                            """)

                # Visualization
                fig = go.Figure()

                # Scatter plot showing spillover effects
                colors = ['red' if t == 0 else 'blue' for t in treatment]

                fig.add_trace(go.Scatter(
                    x=neighbors_treated,
                    y=outcome_with_spillover,
                    mode='markers',
                    marker=dict(color=colors, opacity=0.7),
                    text=[f"Treatment: {t}" for t in treatment],
                    name='Units'
                ))

                fig.update_layout(
                    title="Outcomes vs Neighbor Treatment Rate",
                    xaxis_title="Proportion of Neighbors Treated",
                    yaxis_title="Outcome"
                )

                st.plotly_chart(fig, use_container_width=True)

        elif limitation_category == "External Validity":
            st.markdown("#### 🌍 Generalizability of Causal Findings")

            st.markdown("""
                        <div class="important-box">
                        <h4>🎯 External Validity Concerns</h4>
                        <p>Can we generalize findings from the study sample to other populations, settings, or time periods?</p>
                        </div>
                        """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                            **Threats to External Validity:**

                            1. **Population differences**
                               - Demographics
                               - Baseline characteristics
                               - Selection mechanisms

                            2. **Setting differences**
                               - Geographic location
                               - Institutional context
                               - Time period

                            3. **Treatment implementation**
                               - Dosage/intensity
                               - Delivery method
                               - Compliance patterns

                            4. **Outcome measurement**
                               - Different metrics
                               - Time horizons
                               - Context dependency
                            """)

            with col2:
                st.markdown("#### 🔍 External Validity Assessment Tool")

                # Interactive external validity assessment
                study_population = st.text_input("Study population:", "College students, age 18-22")
                target_population = st.text_input("Target population:", "General adult population")

                study_setting = st.text_input("Study setting:", "University research lab")
                target_setting = st.text_input("Target setting:", "Real-world clinical practice")

                # Similarity assessment
                pop_similarity = st.slider("Population similarity (0-10):", 0, 10, 6)
                setting_similarity = st.slider("Setting similarity (0-10):", 0, 10, 4)
                treatment_similarity = st.slider("Treatment implementation similarity (0-10):", 0, 10, 7)

                # Overall external validity score
                external_validity_score = (pop_similarity + setting_similarity + treatment_similarity) / 3

                if external_validity_score >= 8:
                    validity_assessment = "High external validity"
                    validity_color = "green"
                elif external_validity_score >= 6:
                    validity_assessment = "Moderate external validity"
                    validity_color = "orange"
                else:
                    validity_assessment = "Low external validity"
                    validity_color = "red"

                st.markdown(f"""
                            **External Validity Assessment:**

                            - Population similarity: {pop_similarity}/10
                            - Setting similarity: {setting_similarity}/10
                            - Treatment similarity: {treatment_similarity}/10

                            **Overall Score: {external_validity_score:.1f}/10**

                            **Assessment: <span style="color: {validity_color}; font-weight: bold;">{validity_assessment}</span>**
                            """, unsafe_allow_html=True)

                # Recommendations
                if external_validity_score < 6:
                    st.markdown("""
                                **Recommendations:**
                                - Conduct replication studies in target population
                                - Examine effect heterogeneity
                                - Consider adaptation strategies
                                - Use caution in policy recommendations
                                """)

# Section 9: Summary & References
elif selected_section == "📚 Summary & References":
    st.markdown('<h2 class="section-header">📚 Summary & References</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Key Takeaways", "📖 Further Reading", "🔗 Useful Resources"])

    with tab1:
        st.markdown('<h3 class="subsection-header">📝 Key Takeaways</h3>', unsafe_allow_html=True)

        st.markdown("""
                    <div class="important-box">
                    <h4>🎯 Essential Points to Remember</h4>
                    <p>Core concepts and practical insights from the Rubin Causal Model</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Key concepts summary
        key_concepts = {
            "🔬 Fundamental Framework": [
                "Potential outcomes define what would happen under different treatments",
                "Individual causal effects are differences in potential outcomes",
                "The fundamental problem: we can't observe both potential outcomes",
                "Average treatment effects aggregate individual effects"
            ],
            "⚖️ Critical Assumptions": [
                "SUTVA: No interference between units, consistent treatments",
                "Ignorability: Treatment assignment independent of potential outcomes",
                "Overlap: Positive probability of treatment for all relevant units",
                "These assumptions enable causal identification"
            ],
            "🎯 Estimation Strategies": [
                "Randomized experiments provide the gold standard",
                "Observational studies require additional assumptions",
                "Multiple methods available: matching, IPW, regression, IV",
                "Sensitivity analysis crucial for assessing robustness"
            ],
            "📊 Practical Implementation": [
                "Effect sizes matter more than statistical significance",
                "Consider clinical/practical significance alongside statistical tests",
                "External validity often as important as internal validity",
                "Multiple sources of uncertainty should be acknowledged"
            ]
        }

        for category, points in key_concepts.items():
            st.markdown(f"#### {category}")
            for point in points:
                st.markdown(f"- {point}")
            st.markdown("")

        # Decision flowchart
        st.markdown("#### 🗺️ Decision Flowchart for Causal Analysis")

        # Create a simple decision tree visualization
        flowchart_html = """
                    <div style="text-align: center; font-family: Arial, sans-serif;">
                        <div style="border: 2px solid #1f77b4; padding: 10px; margin: 10px; background-color: #f0f8ff;">
                            <strong>Research Question</strong><br>
                            Is this a causal question?
                        </div>
                        ↓
                        <div style="border: 2px solid #ff7f0e; padding: 10px; margin: 10px; background-color: #fff5f0;">
                            <strong>Study Design</strong><br>
                            Randomized experiment possible?
                        </div>
                        ↓
                        <div style="display: flex; justify-content: space-around;">
                            <div style="border: 2px solid #2ca02c; padding: 10px; margin: 10px; background-color: #f0fff0; width: 45%;">
                                <strong>YES: RCT</strong><br>
                                • Check randomization<br>
                                • Analyze as randomized<br>
                                • Consider compliance
                            </div>
                            <div style="border: 2px solid #d62728; padding: 10px; margin: 10px; background-color: #fff0f0; width: 45%;">
                                <strong>NO: Observational</strong><br>
                                • Identify confounders<br>
                                • Choose method<br>
                                • Check assumptions
                            </div>
                        </div>
                        ↓
                        <div style="border: 2px solid #9467bd; padding: 10px; margin: 10px; background-color: #f8f0ff;">
                            <strong>Interpretation</strong><br>
                            Effect size + Significance + Generalizability
                        </div>
                    </div>
                    """

        st.markdown(flowchart_html, unsafe_allow_html=True)

        # Common pitfalls
        st.markdown("#### ⚠️ Common Pitfalls to Avoid")

        pitfalls = [
            "**Confusing correlation with causation**: Always consider alternative explanations",
            "**Ignoring selection bias**: Treatment assignment is rarely random in observational data",
            "**Over-interpreting p-values**: Statistical significance ≠ practical importance",
            "**Assuming linear effects**: Treatment effects may vary across individuals",
            "**Ignoring external validity**: Findings may not generalize beyond study context",
            "**Multiple testing**: Adjust for multiple comparisons when appropriate",
            "**Cherry-picking results**: Report all analyses, including negative findings"
        ]

        for pitfall in pitfalls:
            st.markdown(f"- {pitfall}")

    with tab2:
        st.markdown('<h3 class="subsection-header">📖 Further Reading</h3>', unsafe_allow_html=True)

        # Essential books
        st.markdown("#### 📚 Essential Books")

        books = [
            {
                "title": "Causal Inference for Statistics, Social, and Biomedical Sciences",
                "authors": "Imbens, G. W., & Rubin, D. B.",
                "year": "2015",
                "description": "Comprehensive treatment of causal inference from the founders of the potential outcomes framework",
                "level": "Advanced"
            },
            {
                "title": "Mostly Harmless Econometrics",
                "authors": "Angrist, J. D., & Pischke, J. S.",
                "year": "2009",
                "description": "Practical guide to empirical research design with emphasis on credible identification",
                "level": "Intermediate"
            },
            {
                "title": "Causal Inference: The Mixtape",
                "authors": "Cunningham, S.",
                "year": "2021",
                "description": "Modern, accessible introduction to causal inference with code examples",
                "level": "Intermediate"
            },
            {
                "title": "The Book of Why",
                "authors": "Pearl, J., & Mackenzie, D.",
                "year": "2018",
                "description": "Accessible introduction to causal thinking for general audiences",
                "level": "Beginner"
            },
            {
                "title": "Counterfactuals and Causal Inference",
                "authors": "Morgan, S. L., & Winship, C.",
                "year": "2015",
                "description": "Sociological perspective on causal analysis methods",
                "level": "Intermediate"
            }
        ]

        for book in books:
            st.markdown(f"""
                        **{book['title']}** ({book['year']})  
                        *{book['authors']}*  
                        Level: {book['level']}  
                        {book['description']}
                        """)
            st.markdown("---")

        # Key papers
        st.markdown("#### 📄 Foundational Papers")

        papers = [
            {
                "title": "Estimating causal effects of treatments in randomized and nonrandomized studies",
                "authors": "Rubin, D. B.",
                "year": "1974",
                "journal": "Journal of Educational Psychology",
                "description": "Original formulation of the Rubin Causal Model"
            },
            {
                "title": "The central role of the propensity score in observational studies for causal effects",
                "authors": "Rosenbaum, P. R., & Rubin, D. B.",
                "year": "1983",
                "journal": "Biometrika",
                "description": "Introduction of propensity score methods"
            },
            {
                "title": "Identification of causal effects using instrumental variables",
                "authors": "Angrist, J. D., Imbens, G. W., & Rubin, D. B.",
                "year": "1996",
                "journal": "Journal of the American Statistical Association",
                "description": "Framework for instrumental variable analysis"
            },
            {
                "title": "Regression discontinuity designs: A guide to practice",
                "authors": "Lee, D. S., & Lemieux, T.",
                "year": "2010",
                "journal": "Journal of Econometrics",
                "description": "Comprehensive guide to regression discontinuity methods"
            }
        ]

        for paper in papers:
            st.markdown(f"""
                        **{paper['title']}** ({paper['year']})  
                        *{paper['authors']}*  
                        {paper['journal']}  
                        {paper['description']}
                        """)
            st.markdown("---")

    with tab3:
        st.markdown('<h3 class="subsection-header">🔗 Useful Resources</h3>', unsafe_allow_html=True)

        # Online courses
        st.markdown("#### 🎓 Online Courses")

        courses = [
            "**Causal Inference** - Brady Neal (YouTube series)",
            "**Causal Diagrams** - University of Pennsylvania (Coursera)",
            "**Econometrics** - MIT OpenCourseWare",
            "**Introduction to Causal Inference** - UC Berkeley",
            "**Statistical Learning** - Stanford Online"
        ]

        for course in courses:
            st.markdown(f"- {course}")

        # Software packages
        st.markdown("#### 💻 Software Packages")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                        **R Packages:**
                        - `MatchIt` - Matching methods
                        - `WeightIt` - Weighting methods
                        - `CausalImpact` - Causal inference using time series
                        - `rdrobust` - Regression discontinuity
                        - `AER` - Applied econometrics
                        - `Matching` - Multivariate matching
                        """)

        with col2:
            st.markdown("""
                        **Python Packages:**
                        - `causalinference` - General causal inference
                        - `dowhy` - Microsoft's causal inference library
                        - `scikit-learn` - Machine learning (some causal methods)
                        - `statsmodels` - Statistical modeling
                        - `linearmodels` - Econometric models
                        - `causalml` - Machine learning for causal inference
                        """)

        # Websites and blogs
        st.markdown("#### 🌐 Websites and Blogs")

        websites = [
            "**Causal Inference Bootcamp** - Stanford Graduate School of Business",
            "**The Effect** - Free online textbook by Nick Huntington-Klein",
            "**Causal Analysis in Theory and Practice** - Judea Pearl's blog",
            "**Almost Surely** - Statistical blog with causal inference content",
            "**Data Colada** - Research methods and statistics blog"
        ]

        for website in websites:
            st.markdown(f"- {website}")

        # Professional organizations
        st.markdown("#### 🏛️ Professional Organizations")

        organizations = [
            "**American Statistical Association (ASA)** - Statistical methods section",
            "**American Economic Association (AEA)** - Econometric methods",
            "**International Biometric Society** - Biostatistical methods",
            "**Society for Research Synthesis Methodology** - Meta-analysis methods",
            "**Academy Health** - Health services research methods"
        ]

        for org in organizations:
            st.markdown(f"- {org}")

        # Final note
        st.markdown("""
                    <div class="important-box">
                    <h4>🎯 Final Note</h4>
                    <p>The Rubin Causal Model provides a powerful framework for thinking about causality, but remember:</p>
                    <ul>
                    <li>No method is perfect - all require untestable assumptions</li>
                    <li>Domain knowledge is crucial for credible causal inference</li>
                    <li>Replication and robustness checks strengthen conclusions</li>
                    <li>Clear communication of limitations is essential</li>
                    </ul>
                    <p><strong>Good causal inference combines rigorous methods with careful reasoning about the substantive problem.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.9em;">
            📊 <strong>Rubin Causal Model: Complete Guide</strong> | 
            Built by Dr Merwan Roudane | 
            For educational purposes
            </div>
            """, unsafe_allow_html=True)