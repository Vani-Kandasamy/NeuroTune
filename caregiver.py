import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import random
import pickle
from io import StringIO
import os

# EEG frequency bands (Hz) and electrode positions
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# EEG electrode positions (Muse headband)
ELECTRODES = ['TP9', 'AF7', 'AF8', 'TP10']

# Expected CSV columns for real EEG dataset
EXPECTED_COLUMNS = [
    'Delta_TP9_mean', 'Delta_TP9_std', 'Delta_AF7_mean', 'Delta_AF7_std', 
    'Delta_AF8_mean', 'Delta_AF8_std', 'Delta_TP10_mean', 'Delta_TP10_std',
    'Theta_TP9_mean', 'Theta_TP9_std', 'Theta_AF7_mean', 'Theta_AF7_std',
    'Theta_AF8_mean', 'Theta_AF8_std', 'Theta_TP10_mean', 'Theta_TP10_std',
    'Alpha_TP9_mean', 'Alpha_TP9_std', 'Alpha_AF7_mean', 'Alpha_AF7_std',
    'Alpha_AF8_mean', 'Alpha_AF8_std', 'Alpha_TP10_mean', 'Alpha_TP10_std',
    'Beta_TP9_mean', 'Beta_TP9_std', 'Beta_AF7_mean', 'Beta_AF7_std',
    'Beta_AF8_mean', 'Beta_AF8_std', 'Beta_TP10_mean', 'Beta_TP10_std',
    'Gamma_TP9_mean', 'Gamma_TP9_std', 'Gamma_AF7_mean', 'Gamma_AF7_std',
    'Gamma_AF8_mean', 'Gamma_AF8_std', 'Gamma_TP10_mean', 'Gamma_TP10_std',
    'Melody #'
]

# Melody categories for ML prediction
MELODY_CATEGORIES = {
    1: "Classical",
    2: "Rock",
    3: "Pop",
    4: "Rap",
    5: "R&B"
}


class IdentityScaler:
    """A simple passthrough scaler used when a fitted scaler is unavailable."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X


def calculate_engagement_score(row):
    """Calculate Engagement Score: Mean(Beta+Gamma) - Mean(Alpha+Theta) across all electrodes"""
    # Calculate mean Beta across all electrodes
    beta_mean = (row['Beta_TP9_mean'] + row['Beta_AF7_mean'] + 
                 row['Beta_AF8_mean'] + row['Beta_TP10_mean']) / 4
    
    # Calculate mean Gamma across all electrodes
    gamma_mean = (row['Gamma_TP9_mean'] + row['Gamma_AF7_mean'] + 
                  row['Gamma_AF8_mean'] + row['Gamma_TP10_mean']) / 4
    
    # Calculate mean Alpha across all electrodes
    alpha_mean = (row['Alpha_TP9_mean'] + row['Alpha_AF7_mean'] + 
                  row['Alpha_AF8_mean'] + row['Alpha_TP10_mean']) / 4
    
    # Calculate mean Theta across all electrodes
    theta_mean = (row['Theta_TP9_mean'] + row['Theta_AF7_mean'] + 
                  row['Theta_AF8_mean'] + row['Theta_TP10_mean']) / 4
    
    # Engagement = Mean(Beta+Gamma) - Mean(Alpha+Theta)
    high_freq = (beta_mean + gamma_mean) / 2
    low_freq = (alpha_mean + theta_mean) / 2
    return high_freq - low_freq

def calculate_focus_score(row):
    """Calculate Focus Score: Theta/Beta Ratio across all electrodes (lower is better focus)"""
    # Calculate mean Beta across all electrodes
    beta_mean = (row['Beta_TP9_mean'] + row['Beta_AF7_mean'] + 
                 row['Beta_AF8_mean'] + row['Beta_TP10_mean']) / 4
    
    # Calculate mean Theta across all electrodes
    theta_mean = (row['Theta_TP9_mean'] + row['Theta_AF7_mean'] + 
                  row['Theta_AF8_mean'] + row['Theta_TP10_mean']) / 4
    
    # Focus = Theta/Beta ratio (lower ratio = better focus)
    if beta_mean == 0:
        return float('inf')  # Very poor focus if no beta activity
    return theta_mean / beta_mean

def calculate_relaxation_score(row):
    """Calculate Relaxation Score: Mean(Alpha+Theta) - Mean(Beta+Gamma) across all electrodes"""
    # Calculate mean Alpha across all electrodes
    alpha_mean = (row['Alpha_TP9_mean'] + row['Alpha_AF7_mean'] + 
                  row['Alpha_AF8_mean'] + row['Alpha_TP10_mean']) / 4
    
    # Calculate mean Theta across all electrodes
    theta_mean = (row['Theta_TP9_mean'] + row['Theta_AF7_mean'] + 
                  row['Theta_AF8_mean'] + row['Theta_TP10_mean']) / 4
    
    # Calculate mean Beta across all electrodes
    beta_mean = (row['Beta_TP9_mean'] + row['Beta_AF7_mean'] + 
                 row['Beta_AF8_mean'] + row['Beta_TP10_mean']) / 4
    
    # Calculate mean Gamma across all electrodes
    gamma_mean = (row['Gamma_TP9_mean'] + row['Gamma_AF7_mean'] + 
                  row['Gamma_AF8_mean'] + row['Gamma_TP10_mean']) / 4
    
    # Relaxation = Mean(Alpha+Theta) - Mean(Beta+Gamma)
    calm_freq = (alpha_mean + theta_mean) / 2
    active_freq = (beta_mean + gamma_mean) / 2
    return calm_freq - active_freq

def process_eeg_scores(df):
    """Process EEG data and calculate cognitive scores"""
    df_processed = df.copy()
    
    # Ensure melody category is available (map from target column if needed)
    if 'melody_category' not in df_processed.columns and 'Melody #' in df_processed.columns:
        df_processed['melody_category'] = df_processed['Melody #']
    
    # Compute aggregate EEG band features across available electrodes
    # Expect columns like 'Delta_TP9_mean', 'Theta_AF7_mean', etc.
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    for band in band_names:
        band_cols = [c for c in df_processed.columns if c.startswith(f"{band}_") and c.endswith("_mean")]
        if band_cols:
            # store as lowercase band name (e.g., 'delta')
            df_processed[band.lower()] = df_processed[band_cols].mean(axis=1)
    
    # Calculate cognitive scores
    df_processed['engagement_score'] = df.apply(calculate_engagement_score, axis=1)
    df_processed['focus_score'] = df.apply(calculate_focus_score, axis=1)
    df_processed['relaxation_score'] = df.apply(calculate_relaxation_score, axis=1)
    
    # Normalize scores to 0-10 scale
    for score_col in ['engagement_score', 'focus_score', 'relaxation_score']:
        min_val = df_processed[score_col].min()
        max_val = df_processed[score_col].max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            # Avoid division by zero; set neutral value 5
            df_processed[f'{score_col}_normalized'] = 5.0
        else:
            df_processed[f'{score_col}_normalized'] = 10 * (df_processed[score_col] - min_val) / (max_val - min_val)
    
    return df_processed


def load_default_model():
    """Load default pre-trained Random Forest model from pickle file"""
    model_path = "best_RF_with_time"  # Default model file name
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            # If pickle contains dictionary with model and other info
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            feature_names = model_data.get('feature_names', [])
            accuracy = model_data.get('accuracy', 0.0)
        else:
            # If pickle contains just the model
            model = model_data
            scaler = None
            feature_names = []
            accuracy = 0.0
        
        # If scaler is missing or not fitted, fall back to IdentityScaler
        if scaler is None:
            scaler = IdentityScaler()
        else:
            # Try a light check to see if scaler is fitted; if not, replace
            try:
                _ = scaler.transform(np.zeros((1, getattr(scaler, 'n_features_in_', 1))))
            except Exception:
                scaler = IdentityScaler()
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'model_type': 'pre_trained',
            'loaded_from_file': True
        }
    except FileNotFoundError:
        # If default model file doesn't exist, return None to use trained model
        return None
    except Exception as e:
        st.warning(f"Error loading default model: {str(e)}. Will train new model on data.")
        return None

def initialize_caregiver_session_state():
    """Initialize session state for caregiver dashboard"""
    if 'processed_eeg_data' not in st.session_state:
        st.session_state.processed_eeg_data = None
    
    if 'ml_model_results' not in st.session_state:
        # Try to load default pre-trained model
        model_data = load_default_model()
        if model_data:
            st.session_state.ml_model_results = model_data
            st.success("✅ Loaded default pre-trained model 'random_forest_model.pkl'")
        else:
            st.session_state.ml_model_results = None

def patient_selector():
    """Patient selection widget"""
    df = st.session_state.processed_eeg_data
    
    if df is None or df.empty:
        st.info("No data available")
        st.session_state.selected_patient = None
        return pd.DataFrame()
    
    patients = sorted(df['user_id'].unique())
    
    selected = st.selectbox(
        "Select Patient",
        ["All Patients"] + list(patients),
        index=0
    )
    
    if selected != "All Patients":
        st.session_state.selected_patient = selected
        return df[df['user_id'] == selected]
    else:
        st.session_state.selected_patient = None
        return df

def ml_model_dashboard():
    """Display ML model performance and insights"""
    st.subheader("🤖 ML Model Performance")
    
    results = st.session_state.ml_model_results
    
    if not results:
        st.warning("No ML model available. Please ensure 'random_forest_model.pkl' is in the project directory.")
        return
    
    if results.get('loaded_from_file', False):
        st.success("✅ **Using Pre-trained Model** - `random_forest_model.pkl`")
        
        # Show model capabilities for pre-trained model
        st.markdown("""
        **Model Capabilities:**
        - ✅ Music genre prediction (Classical, Rock, Pop, Rap, R&B)
        - ✅ EEG-based cognitive score analysis  
        - ✅ Patient-specific recommendations
        - ✅ Real-time inference on new data
        """)
        
        # Model info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = results.get('accuracy', 0)
            st.metric("Model Accuracy", f"{accuracy:.1%}" if accuracy > 0 else "Pre-trained")
        
        with col2:
            st.metric("Model Type", "Random Forest")
        
        with col3:
            st.metric("Features", "40 EEG + 3 Cognitive")
        
        st.info("📊 **Ready for Predictions**: Upload EEG data to get music genre predictions and cognitive insights.")
        
    else:
        st.error("❌ **No Model Available**: Please place your trained 'random_forest_model.pkl' file in the project directory.")

def cognitive_insights_dashboard(df):
    """Display cognitive and emotional insights"""
    st.subheader("🧠 Cognitive & Emotional Insights")
    
    # Overall cognitive metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_engagement = df['engagement_score_normalized'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.1f}/10", 
                 delta=f"{avg_engagement - 5:.1f}" if avg_engagement != 5 else None)
    
    with col2:
        avg_focus = 10 - df['focus_score_normalized'].mean()  # Inverse for better interpretation
        st.metric("Avg Focus", f"{avg_focus:.1f}/10",
                 delta=f"{avg_focus - 5:.1f}" if avg_focus != 5 else None)
    
    with col3:
        avg_relaxation = df['relaxation_score_normalized'].mean()
        st.metric("Avg Relaxation", f"{avg_relaxation:.1f}/10",
                 delta=f"{avg_relaxation - 5:.1f}" if avg_relaxation != 5 else None)
    
    # Cognitive scores by melody category
    st.subheader("📈 Cognitive Response by Melody Type")
    
    # Ensure melody_category exists
    if 'melody_category' not in df.columns and 'Melody #' in df.columns:
        df = df.copy()
        df['melody_category'] = df['Melody #']
    
    category_analysis = df.groupby('melody_category').agg({
        'engagement_score_normalized': 'mean',
        'focus_score_normalized': 'mean', 
        'relaxation_score_normalized': 'mean'
    }).reset_index()
    
    category_analysis['melody_type'] = category_analysis['melody_category'].map(MELODY_CATEGORIES)
    category_analysis['focus_score_normalized'] = 10 - category_analysis['focus_score_normalized']  # Inverse
    
    # Create subplot for all three scores
    fig_cognitive = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Engagement', 'Focus', 'Relaxation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (score, title) in enumerate([
        ('engagement_score_normalized', 'Engagement'),
        ('focus_score_normalized', 'Focus'), 
        ('relaxation_score_normalized', 'Relaxation')
    ]):
        fig_cognitive.add_trace(
            go.Bar(
                x=category_analysis['melody_type'],
                y=category_analysis[score],
                name=title,
                marker_color=colors[i],
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig_cognitive.update_layout(height=400, title_text="Cognitive Scores by Melody Category")
    fig_cognitive.update_yaxes(range=[0, 10])
    st.plotly_chart(fig_cognitive, use_container_width=True)
    
    # EEG band analysis by electrode
    st.subheader("🌊 EEG Frequency Band Analysis by Electrode")
    
    # Calculate average power for each band across all electrodes
    band_electrode_data = []
    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
        for electrode in ELECTRODES:
            col_name = f'{band}_{electrode}_mean'
            if col_name in df.columns:
                avg_power = df[col_name].mean()
                band_electrode_data.append({
                    'Band': band,
                    'Electrode': electrode,
                    'Average_Power': avg_power
                })
    
    if band_electrode_data:
        band_df = pd.DataFrame(band_electrode_data)
        
        # Create heatmap showing band power by electrode
        pivot_df = band_df.pivot(index='Band', columns='Electrode', values='Average_Power')
        
        fig_heatmap = px.imshow(
            pivot_df.values,
            labels=dict(x="Electrode Position", y="Frequency Band", color="Power (μV²)"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='viridis',
            title="EEG Band Power Heatmap by Electrode Position"
        )
        
        # Add text annotations
        for i, band in enumerate(pivot_df.index):
            for j, electrode in enumerate(pivot_df.columns):
                fig_heatmap.add_annotation(
                    x=j, y=i,
                    text=f"{pivot_df.iloc[i, j]:.1f}",
                    showarrow=False,
                    font=dict(color="white", size=10)
                )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Bar chart comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate distribution of melody preferences
            if 'Melody #' in df.columns:
                melody_dist = df['Melody #'].value_counts().sort_index()
                fig_melody = px.bar(
                    x=[MELODY_CATEGORIES.get(int(i), str(i)) for i in melody_dist.index],
                    y=melody_dist.values,
                    title="🎵 Music Genre Preference Distribution",
                    labels={'x': 'Music Genre', 'y': 'Count'},
                    color=melody_dist.values,
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig_melody, use_container_width=True)
            else:
                st.info("No 'Melody #' column available for distribution plot.")
        
        with col2:
            # Average by electrode
            electrode_avg = band_df.groupby('Electrode')['Average_Power'].mean().reset_index()
            fig_electrodes = px.bar(
                electrode_avg, x='Electrode', y='Average_Power',
                title="Average Power by Electrode Position",
                labels={'Average_Power': 'Power (μV²)'},
                color='Average_Power',
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_electrodes, use_container_width=True)
    else:
        st.info("No electrode-specific EEG data available for visualization.")

def patient_specific_analysis(patient_df):
    """Detailed analysis for a specific patient"""
    st.subheader(f"👤 Patient Analysis: {st.session_state.selected_patient}")
    
    # Patient summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sessions = len(patient_df)
        st.metric("Total Sessions", sessions)
    
    with col2:
        total_time = patient_df['duration_seconds'].sum() / 60
        st.metric("Total Time", f"{total_time:.0f} min")
    
    with col3:
        preferred_category = patient_df['melody_category'].mode().iloc[0]
        st.metric("Preferred Type", MELODY_CATEGORIES[preferred_category])
    
    with col4:
        avg_engagement = patient_df['engagement_score_normalized'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.1f}/10")
    
    # Timeline analysis
    st.subheader("📅 Progress Over Time")
    
    patient_df_sorted = patient_df.sort_values('timestamp')
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=patient_df_sorted['timestamp'],
        y=patient_df_sorted['engagement_score_normalized'],
        mode='lines+markers',
        name='Engagement',
        line=dict(color='#FF6B6B')
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=patient_df_sorted['timestamp'],
        y=10 - patient_df_sorted['focus_score_normalized'],
        mode='lines+markers',
        name='Focus',
        line=dict(color='#4ECDC4')
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=patient_df_sorted['timestamp'],
        y=patient_df_sorted['relaxation_score_normalized'],
        mode='lines+markers',
        name='Relaxation',
        line=dict(color='#45B7D1')
    ))
    
    fig_timeline.update_layout(
        title="Cognitive Scores Over Time",
        xaxis_title="Date",
        yaxis_title="Score (0-10)",
        yaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Melody recommendations for this patient
    st.subheader("🎵 ML-Based Melody Recommendations")
    
    # Use the trained model to predict best melody category for this patient
    model_results = st.session_state.ml_model_results
    if not model_results or 'model' not in model_results or model_results['model'] is None:
        st.warning("ML model not available for recommendations. Please add 'random_forest_model.pkl'.")
        return
    
    # Get average EEG features for this patient
    feature_columns = ['delta', 'theta', 'alpha', 'beta', 'gamma', 
                      'engagement_score', 'focus_score', 'relaxation_score']
    # Ensure all required features exist
    missing_features = [c for c in feature_columns if c not in patient_df.columns]
    if missing_features:
        st.warning(f"Missing features for prediction: {missing_features}")
        return
    patient_features = patient_df[feature_columns].mean().values.reshape(1, -1)
    # Apply scaler (IdentityScaler if real scaler not available)
    patient_features_scaled = model_results['scaler'].transform(patient_features)
    
    # Predict probabilities for each melody category
    probabilities = model_results['model'].predict_proba(patient_features_scaled)[0]
    
    recommendations = []
    for i, prob in enumerate(probabilities):
        recommendations.append({
            'category': MELODY_CATEGORIES[i],
            'probability': prob,
            'recommendation_strength': 'High' if prob > 0.5 else 'Medium' if prob > 0.3 else 'Low'
        })
    
    recommendations.sort(key=lambda x: x['probability'], reverse=True)
    
    for rec in recommendations:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{rec['category']} Melodies**")
        with col2:
            st.write(f"{rec['probability']:.1%}")
        with col3:
            color = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}[rec['recommendation_strength']]
            st.write(f"{color} {rec['recommendation_strength']}")

def export_patient_report(df):
    """Export detailed patient analysis report"""
    st.subheader("📊 Export Patient Reports")
    
    if st.session_state.selected_patient:
        patient_df = df[df['user_id'] == st.session_state.selected_patient]
        patient_id = st.session_state.selected_patient
    else:
        patient_df = df
        patient_id = "All_Patients"
    
    # Generate comprehensive report
    report = {
        'patient_id': patient_id,
        'report_date': datetime.now().isoformat(),
        'summary': {
            'total_sessions': len(patient_df),
            'total_duration_minutes': patient_df['duration_seconds'].sum() / 60,
            'average_engagement': patient_df['engagement_score_normalized'].mean(),
            'average_focus': (10 - patient_df['focus_score_normalized']).mean(),
            'average_relaxation': patient_df['relaxation_score_normalized'].mean(),
            'preferred_melody_category': MELODY_CATEGORIES[patient_df['melody_category'].mode().iloc[0]]
        },
        'eeg_analysis': {
            'delta_avg': patient_df['delta'].mean(),
            'theta_avg': patient_df['theta'].mean(),
            'alpha_avg': patient_df['alpha'].mean(),
            'beta_avg': patient_df['beta'].mean(),
            'gamma_avg': patient_df['gamma'].mean()
        },
        'cognitive_scores_by_category': patient_df.groupby('melody_category').agg({
            'engagement_score_normalized': 'mean',
            'focus_score_normalized': 'mean',
            'relaxation_score_normalized': 'mean'
        }).to_dict(),
        'ml_model_accuracy': (st.session_state.ml_model_results.get('accuracy')
                              if st.session_state.get('ml_model_results') else None),
        'recommendations': "Based on EEG patterns, focus on melodies that enhance target cognitive states"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download JSON Report"):
            st.download_button(
                label="Download Report",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"patient_analysis_{patient_id}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download Raw Data CSV"):
            csv_data = patient_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"eeg_data_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def caregiver_dashboard():
    """Main caregiver dashboard with ML and EEG analysis"""
    initialize_caregiver_session_state()
    
    user_info = st.session_state.user_info
    
    st.title("👩‍⚕️ Caregiver ML Analytics Dashboard")
    st.markdown(f"Welcome back, **{user_info['name']}**! Advanced EEG analysis and ML-powered insights.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 ML Analytics")
        page = st.selectbox(
            "Select Analysis",
            ["Overview", "ML Model Performance", "Patient Analysis", "Cognitive Insights", "EEG Data Upload", "Export Reports"]
        )
        
        st.markdown("---")
        st.markdown("### 👤 Patient Selection")
        df_filtered = patient_selector()
        
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if page == "Overview":
        st.subheader("📊 System Overview")
        
        df = st.session_state.processed_eeg_data
        
        if df is not None and not df.empty:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_patients = df['user_id'].nunique()
                st.metric("Total Patients", total_patients)
            
            with col2:
                total_sessions = len(df)
                st.metric("Total Sessions", total_sessions)
            
            with col3:
                model_accuracy = st.session_state.ml_model_results.get('accuracy', 0) if st.session_state.ml_model_results else 0
                st.metric("ML Model Accuracy", f"{model_accuracy:.1%}" if model_accuracy > 0 else "No Model")
        
            with col4:
                avg_engagement = df['engagement_score_normalized'].mean()
                st.metric("Avg Engagement", f"{avg_engagement:.1f}/10")
        else:
            st.info("📤 **No data loaded yet.** Please upload a CSV file using the 'EEG Data Upload' section to begin analysis.")
        
        st.markdown("---")
        
        # Recent activity and insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Recent Trends")
            
            # Last 30 days trend
            recent_data = df[df['timestamp'] >= datetime.now() - timedelta(days=30)]
            daily_engagement = recent_data.groupby(recent_data['timestamp'].dt.date)['engagement_score_normalized'].mean()
            
            if not daily_engagement.empty:
                fig_trend = px.line(
                    x=daily_engagement.index,
                    y=daily_engagement.values,
                    title="30-Day Engagement Trend",
                    labels={'x': 'Date', 'y': 'Avg Engagement Score'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No recent data available")
        
        with col2:
            st.subheader("🎯 Top Performing Melodies / Categories")
            if df is not None and not df.empty:
                if 'melody_name' in df.columns:
                    top = df.groupby(['melody_category', 'melody_name'])['engagement_score_normalized'].mean().nlargest(5)
                    melody_data = []
                    for (category, name), score in top.items():
                        melody_data.append({
                            'melody': name,
                            'category': MELODY_CATEGORIES.get(category, str(category)),
                            'avg_score': score
                        })
                    for melody in melody_data:
                        st.markdown(f"**{melody['melody']}** ({melody['category']})  \nAvg Score: {melody['avg_score']:.1f}/10")
                else:
                    top = df.groupby(['melody_category'])['engagement_score_normalized'].mean().nlargest(5)
                    for category, score in top.items():
                        st.markdown(f"**{MELODY_CATEGORIES.get(category, str(category))}**  \nAvg Score: {score:.1f}/10")
            else:
                st.info("No data available yet.")
    
    elif page == "ML Model Performance":
        ml_model_dashboard()
    
    elif page == "Patient Analysis":
        if st.session_state.selected_patient:
            patient_df = df_filtered
            patient_specific_analysis(patient_df)
        else:
            st.info("Please select a specific patient from the sidebar to view detailed analysis.")
            cognitive_insights_dashboard(df_filtered)
    
    elif page == "Cognitive Insights":
        cognitive_insights_dashboard(df_filtered)
    
    elif page == "EEG Data Upload":
        st.subheader("📤 Upload Patient EEG Data")
        
        # Show current model status
        model_results = st.session_state.ml_model_results
        if model_results and model_results.get('loaded_from_file', False):
            st.info("🤖 **Using Default Pre-trained Model** - `random_forest_model.pkl`")
        else:
            st.warning("⚠️ **No Model Available** - Please ensure `random_forest_model.pkl` is in the project directory")
        
        st.markdown("---")
        
        # Patient selection/creation interface
        st.subheader("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_option = st.radio(
                "Select Patient Option:",
                ["Existing Patient", "New Patient"]
            )
        
        with col2:
            if patient_option == "Existing Patient":
                # Get list of existing patients from current data
                existing_patients = []
                if st.session_state.processed_eeg_data is not None and not st.session_state.processed_eeg_data.empty:
                    existing_patients = sorted(st.session_state.processed_eeg_data['user_id'].unique())
                
                if existing_patients:
                    selected_patient = st.selectbox("Select Existing Patient:", existing_patients)
                else:
                    st.info("No existing patients found. Please create a new patient.")
                    selected_patient = None
            else:
                selected_patient = st.text_input("Enter New Patient ID:", placeholder="e.g., P001, PATIENT_001")
        
        if (patient_option == "Existing Patient" and selected_patient) or (patient_option == "New Patient" and selected_patient):
            st.markdown("---")
            st.markdown(f"**Uploading data for Patient: {selected_patient}**")
            
            st.markdown("""
            **Required CSV Format:**
            - **Frequency Bands per Electrode**: `Delta_TP9_mean`, `Delta_TP9_std`, `Delta_AF7_mean`, etc.
            - **Electrodes**: TP9, AF7, AF8, TP10 (Muse headband positions)
            - **Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma
            - **Target**: `Melody #` (1=Classical, 2=Rock, 3=Pop, 4=Rap, 5=R&B)
            - **Note**: Patient ID will be automatically assigned - do not include `user_id` column
            
            **Expected Format Example:**
            ```
            Delta_TP9_mean, Delta_TP9_std, Delta_AF7_mean, Delta_AF7_std, ..., Melody #
            15.2, 3.1, 14.8, 2.9, ..., 1
            ```
            """)
            
            uploaded_file = st.file_uploader(f"Choose EEG CSV file for {selected_patient}", type="csv", key="patient_data_upload")
        
            if uploaded_file is not None and selected_patient:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    
                    # Validate required columns for the new format
                    required_cols = []
                    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                        for electrode in ELECTRODES:
                            required_cols.append(f'{band}_{electrode}_mean')
                    required_cols.append('Melody #')
                    
                    missing_cols = [col for col in required_cols if col not in new_data.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                        st.info("Expected columns format:")
                        st.code(", ".join(required_cols[:10]) + ", ...")
                    else:
                        # Set patient-specific metadata (user_id not expected in CSV)
                        new_data['user_id'] = selected_patient
                        
                        # Add other metadata columns if not present
                        if 'timestamp' not in new_data.columns:
                            new_data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(new_data), freq='1H')
                        if 'session_id' not in new_data.columns:
                            new_data['session_id'] = [f"{selected_patient}_S{i:04d}" for i in range(len(new_data))]
                        
                        st.info(f"✅ **Patient ID assigned**: All records tagged with Patient ID '{selected_patient}'")
                        
                        # Process the new data
                        processed_new_data = process_eeg_scores(new_data)
                        
                        # Update session state - append to existing data
                        if st.session_state.processed_eeg_data is not None:
                            # Remove existing data for this patient if updating
                            existing_data = st.session_state.processed_eeg_data[
                                st.session_state.processed_eeg_data['user_id'] != selected_patient
                            ]
                            st.session_state.processed_eeg_data = pd.concat([
                                existing_data, 
                                processed_new_data
                            ]).reset_index(drop=True)
                        else:
                            st.session_state.processed_eeg_data = processed_new_data
                        
                        # Data processed successfully - model predictions available if default model loaded
                        if st.session_state.ml_model_results and st.session_state.ml_model_results.get('loaded_from_file', False):
                            st.info("✅ Data processed successfully. Using default pre-trained model for predictions.")
                        else:
                            st.warning("⚠️ No pre-trained model available. Please ensure 'random_forest_model.pkl' is in the project directory for ML predictions.")
                        
                        st.success(f"Successfully uploaded and processed {len(new_data)} records for Patient {selected_patient}!")
                        
                        # Show patient-specific data summary
                        patient_data = st.session_state.processed_eeg_data[
                            st.session_state.processed_eeg_data['user_id'] == selected_patient
                        ]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records for Patient", len(patient_data))
                        with col2:
                            st.metric("Total Patients", st.session_state.processed_eeg_data['user_id'].nunique())
                        with col3:
                            if st.session_state.ml_model_results and st.session_state.ml_model_results.get('accuracy'):
                                st.metric("Model Accuracy", f"{st.session_state.ml_model_results['accuracy']:.1%}")
                            else:
                                st.metric("Model Status", "Ready")
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file matches the expected format with electrode-specific EEG data.")
        
        # Show expected format
        st.subheader("📋 Expected Data Format")
        sample_cols = ['Delta_TP9_mean', 'Delta_TP9_std', 'Theta_TP9_mean', 'Alpha_TP9_mean', 'Beta_TP9_mean', 'Gamma_TP9_mean', 'Melody #']
        sample_data = pd.DataFrame({
            'Delta_TP9_mean': [15.2, 14.8, 16.1],
            'Delta_TP9_std': [3.1, 2.9, 3.4],
            'Theta_TP9_mean': [12.5, 11.8, 13.2],
            'Alpha_TP9_mean': [18.7, 19.2, 17.9],
            'Beta_TP9_mean': [14.3, 15.1, 13.8],
            'Gamma_TP9_mean': [8.2, 7.9, 8.5],
            'Melody #': [1, 2, 3]
        })
        st.dataframe(sample_data)
        
        # Show current patients summary
        st.subheader("📊 Patient Data Summary")
        if st.session_state.processed_eeg_data is not None and not st.session_state.processed_eeg_data.empty:
            # Patient summary table
            patient_summary = st.session_state.processed_eeg_data.groupby('user_id').agg({
                'engagement_score_normalized': 'mean',
                'focus_score_normalized': 'mean', 
                'relaxation_score_normalized': 'mean',
                'Melody #': 'count'
            }).round(2)
            patient_summary.columns = ['Avg Engagement', 'Avg Focus', 'Avg Relaxation', 'Total Records']
            st.dataframe(patient_summary)
        else:
            st.info("No patient data loaded yet. Please upload CSV files for individual patients to begin analysis.")
    
    elif page == "Export Reports":
        export_patient_report(df_filtered)
