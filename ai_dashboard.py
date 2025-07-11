"""
AI Model Performance Dashboard

Interactive dashboard for monitoring AI dimensioning model health, accuracy, and performance.
Provides comprehensive insights, visualizations, and model analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.infrastructure.ai.ai_trainer_nn import AdvancedDimensionAITrainer
    from src.infrastructure.ai.dimension_ai import DWGAnalyzer
    from train_ai_models import AIModelTrainer
except ImportError as e:
    st.error(f"Failed to import AI modules: {e}")
    st.error("Please ensure the AI modules are properly installed.")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="AI Dimensioning Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .info-metric {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

class AIModelDashboard:
    """Main dashboard class for AI model analytics."""
    
    def __init__(self, storage_path: str = "ai_training_data"):
        self.storage_path = Path(storage_path)
        self.trainer = None
        self.analyzer = None
        self.model_stats = None
        self.data_stats = None
        
    def initialize_components(self):
        """Initialize AI components."""
        try:
            self.trainer = AIModelTrainer(storage_path=str(self.storage_path))
            self.analyzer = DWGAnalyzer(storage_path=str(self.storage_path))
            return True
        except Exception as e:
            st.error(f"Failed to initialize AI components: {e}")
            return False
    
    def load_model_statistics(self):
        """Load comprehensive model statistics."""
        try:
            stats = self.trainer.get_training_statistics()
            self.model_stats = stats.get('models', {})
            self.data_stats = stats.get('data', {})
            return True
        except Exception as e:
            st.error(f"Failed to load model statistics: {e}")
            return False
    
    def load_training_data(self):
        """Load training data from database."""
        try:
            db_path = self.storage_path / "training_data.db"
            if not db_path.exists():
                return None, None, None
            
            conn = sqlite3.connect(db_path)
            
            # Load training examples
            examples_df = pd.read_sql_query("""
                SELECT * FROM training_examples
                ORDER BY timestamp DESC
            """, conn)
            
            # Load geometry features
            geometry_df = pd.read_sql_query("""
                SELECT * FROM geometry_features
            """, conn)
            
            # Load dimension features
            dimensions_df = pd.read_sql_query("""
                SELECT * FROM dimension_features
            """, conn)
            
            conn.close()
            return examples_df, geometry_df, dimensions_df
            
        except Exception as e:
            st.error(f"Failed to load training data: {e}")
            return None, None, None
    
    def get_model_health_status(self):
        """Determine overall model health status."""
        if not self.model_stats or 'training_metadata' not in self.model_stats:
            return "No Model", "danger", "No trained model found"
        
        metadata = self.model_stats['training_metadata']
        if 'performance_metrics' not in metadata:
            return "Unknown", "warning", "Performance metrics not available"
        
        entity_perf = metadata['performance_metrics']['entity_importance']
        accuracy = entity_perf.get('accuracy', 0)
        ensemble_accuracy = entity_perf.get('ensemble_accuracy', 0)
        
        if ensemble_accuracy >= 0.95:
            return "Excellent", "success", f"Ensemble accuracy: {ensemble_accuracy:.1%}"
        elif ensemble_accuracy >= 0.90:
            return "Good", "success", f"Ensemble accuracy: {ensemble_accuracy:.1%}"
        elif ensemble_accuracy >= 0.80:
            return "Fair", "warning", f"Ensemble accuracy: {ensemble_accuracy:.1%}"
        else:
            return "Poor", "danger", f"Ensemble accuracy: {ensemble_accuracy:.1%}"
    
    def create_accuracy_gauge(self, accuracy: float, title: str):
        """Create a gauge chart for accuracy metrics."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = accuracy * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 90, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 95], 'color': "lightgreen"},
                    {'range': [95, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_training_progress_chart(self, history: dict):
        """Create training progress visualization."""
        if not history:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training/Validation Accuracy', 'Training/Validation Loss',
                          'Learning Rate', 'F1 Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history.get('accuracy', [])) + 1))
        
        # Accuracy plot
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy', 
                          line=dict(color='blue')),
                row=1, col=1
            )
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy',
                          line=dict(color='red')),
                row=1, col=1
            )
        
        # Loss plot
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['loss'], name='Training Loss',
                          line=dict(color='green')),
                row=1, col=2
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss',
                          line=dict(color='orange')),
                row=1, col=2
            )
        
        # Learning rate (if available)
        if 'lr' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['lr'], name='Learning Rate',
                          line=dict(color='purple')),
                row=2, col=1
            )
        
        # F1 Score (if available)
        if 'f1_score' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['f1_score'], name='F1 Score',
                          line=dict(color='brown')),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Training Progress Analytics")
        return fig
    
    def create_entity_distribution_chart(self, geometry_df):
        """Create entity type distribution visualization."""
        if geometry_df is None or geometry_df.empty:
            return None
        
        entity_counts = geometry_df['entity_type'].value_counts()
        
        fig = px.pie(
            values=entity_counts.values,
            names=entity_counts.index,
            title="Entity Type Distribution in Training Data",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        return fig
    
    def create_layer_analysis_chart(self, geometry_df):
        """Create layer analysis visualization."""
        if geometry_df is None or geometry_df.empty:
            return None
        
        # Get top 15 layers by entity count
        layer_counts = geometry_df['layer_name'].value_counts().head(15)
        
        fig = px.bar(
            x=layer_counts.index,
            y=layer_counts.values,
            title="Top 15 Layers by Entity Count",
            labels={'x': 'Layer Name', 'y': 'Entity Count'},
            color=layer_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig
    
    def create_dimension_analysis_chart(self, dimensions_df):
        """Create dimension analysis visualization."""
        if dimensions_df is None or dimensions_df.empty:
            return None
        
        # Dimension type distribution
        dim_counts = dimensions_df['dimension_type'].value_counts()
        
        fig = px.bar(
            x=dim_counts.index,
            y=dim_counts.values,
            title="Dimension Type Distribution",
            labels={'x': 'Dimension Type', 'y': 'Count'},
            color=dim_counts.values,
            color_continuous_scale='blues'
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_training_timeline(self, examples_df):
        """Create training data timeline."""
        if examples_df is None or examples_df.empty:
            return None
        
        # Convert timestamp to datetime
        examples_df['timestamp'] = pd.to_datetime(examples_df['timestamp'])
        examples_df['date'] = examples_df['timestamp'].dt.date
        
        # Group by date
        daily_counts = examples_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Training Data Collection Timeline",
            labels={'date': 'Date', 'count': 'Files Analyzed'},
            markers=True
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_confusion_matrix(self, confusion_matrix_data):
        """Create confusion matrix heatmap."""
        if not confusion_matrix_data:
            return None
        
        labels = ['Critical', 'Important', 'Detail', 'Ignore']
        
        fig = px.imshow(
            confusion_matrix_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            title="Entity Importance Classification - Confusion Matrix",
            color_continuous_scale='Blues'
        )
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(confusion_matrix_data[i][j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix_data[i][j] > confusion_matrix_data[i][j] * 0.5 else "black")
                )
        
        fig.update_layout(height=400)
        return fig
    
    def display_model_recommendations(self):
        """Display model improvement recommendations."""
        if not self.model_stats or 'training_metadata' not in self.model_stats:
            st.warning("No model statistics available for recommendations.")
            return
        
        metadata = self.model_stats['training_metadata']
        recommendations = []
        
        # Check accuracy
        if 'performance_metrics' in metadata:
            entity_perf = metadata['performance_metrics']['entity_importance']
            accuracy = entity_perf.get('accuracy', 0)
            ensemble_accuracy = entity_perf.get('ensemble_accuracy', 0)
            
            if ensemble_accuracy < 0.90:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Low Accuracy Detected',
                    'message': f'Ensemble accuracy is {ensemble_accuracy:.1%}. Consider adding more training data or retraining with different hyperparameters.'
                })
            elif ensemble_accuracy < 0.95:
                recommendations.append({
                    'type': 'info',
                    'title': 'Good Accuracy',
                    'message': f'Ensemble accuracy is {ensemble_accuracy:.1%}. Consider fine-tuning with more diverse examples for optimal performance.'
                })
            else:
                recommendations.append({
                    'type': 'success',
                    'title': 'Excellent Accuracy',
                    'message': f'Ensemble accuracy is {ensemble_accuracy:.1%}. Model is performing excellently and ready for production use.'
                })
        
        # Check training data volume
        if self.data_stats:
            total_examples = self.data_stats.get('total_examples', 0)
            
            if total_examples < 50:
                recommendations.append({
                    'type': 'danger',
                    'title': 'Insufficient Training Data',
                    'message': f'Only {total_examples} training examples. Recommend at least 100 DWG files for robust training.'
                })
            elif total_examples < 200:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Limited Training Data',
                    'message': f'{total_examples} training examples. Adding more diverse DWG files will improve model robustness.'
                })
        
        # Check model age
        if 'training_date' in metadata:
            training_date = datetime.fromisoformat(metadata['training_date'])
            days_old = (datetime.now() - training_date).days
            
            if days_old > 90:
                recommendations.append({
                    'type': 'info',
                    'title': 'Model Age',
                    'message': f'Model was trained {days_old} days ago. Consider retraining with recent project data for optimal performance.'
                })
        
        # Display recommendations
        if recommendations:
            st.subheader("🎯 Model Improvement Recommendations")
            
            for rec in recommendations:
                if rec['type'] == 'success':
                    st.success(f"✅ {rec['title']}: {rec['message']}")
                elif rec['type'] == 'info':
                    st.info(f"ℹ️ {rec['title']}: {rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"⚠️ {rec['title']}: {rec['message']}")
                elif rec['type'] == 'danger':
                    st.error(f"🚨 {rec['title']}: {rec['message']}")
    
    def render_dashboard(self):
        """Render the complete dashboard."""
        st.markdown('<div class="main-header">🧠 AI Dimensioning Model Dashboard</div>', unsafe_allow_html=True)
        
        # Initialize components
        if not self.initialize_components():
            st.error("Failed to initialize dashboard components. Please check your AI model installation.")
            return
        
        # Load statistics
        if not self.load_model_statistics():
            st.error("Failed to load model statistics. Please ensure models are trained.")
            return
        
        # Sidebar configuration
        st.sidebar.header("📊 Dashboard Settings")
        
        # Storage path selector
        storage_path = st.sidebar.text_input(
            "Storage Path",
            value=str(self.storage_path),
            help="Path to AI training data and models"
        )
        
        if storage_path != str(self.storage_path):
            self.storage_path = Path(storage_path)
            st.experimental_rerun()
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Model health overview
        st.header("🏥 Model Health Overview")
        
        health_status, health_color, health_message = self.get_model_health_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Status", health_status, help=health_message)
        
        with col2:
            if self.data_stats:
                st.metric("Training Examples", self.data_stats.get('total_examples', 0))
        
        with col3:
            if self.data_stats:
                st.metric("Total Entities", self.data_stats.get('total_entities', 0))
        
        with col4:
            if self.data_stats:
                st.metric("Dimensioned Entities", self.data_stats.get('total_dimensioned_entities', 0))
        
        # Performance metrics
        if self.model_stats and 'training_metadata' in self.model_stats:
            metadata = self.model_stats['training_metadata']
            
            if 'performance_metrics' in metadata:
                st.header("📈 Performance Metrics")
                
                entity_perf = metadata['performance_metrics']['entity_importance']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    accuracy = entity_perf.get('accuracy', 0)
                    fig_acc = self.create_accuracy_gauge(accuracy, "Model Accuracy")
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    ensemble_accuracy = entity_perf.get('ensemble_accuracy', 0)
                    fig_ens = self.create_accuracy_gauge(ensemble_accuracy, "Ensemble Accuracy")
                    st.plotly_chart(fig_ens, use_container_width=True)
                
                with col3:
                    val_accuracy = entity_perf.get('val_accuracy', 0)
                    fig_val = self.create_accuracy_gauge(val_accuracy, "Validation Accuracy")
                    st.plotly_chart(fig_val, use_container_width=True)
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{entity_perf.get('precision', 0):.2%}")
                
                with col2:
                    st.metric("Recall", f"{entity_perf.get('recall', 0):.2%}")
                
                with col3:
                    st.metric("F1 Score", f"{entity_perf.get('f1_score', 0):.2%}")
                
                with col4:
                    st.metric("Validation Loss", f"{entity_perf.get('val_loss', 0):.4f}")
        
        # Training progress
        if self.model_stats and 'training_metadata' in self.model_stats:
            metadata = self.model_stats['training_metadata']
            
            if 'performance_metrics' in metadata:
                entity_perf = metadata['performance_metrics']['entity_importance']
                
                if 'training_history' in entity_perf:
                    st.header("📊 Training Progress")
                    
                    history = entity_perf['training_history']
                    fig_progress = self.create_training_progress_chart(history)
                    
                    if fig_progress:
                        st.plotly_chart(fig_progress, use_container_width=True)
                
                # Confusion matrix
                if 'confusion_matrix' in entity_perf:
                    st.header("🎯 Classification Analysis")
                    
                    confusion_matrix_data = entity_perf['confusion_matrix']
                    fig_confusion = self.create_confusion_matrix(confusion_matrix_data)
                    
                    if fig_confusion:
                        st.plotly_chart(fig_confusion, use_container_width=True)
        
        # Training data analysis
        examples_df, geometry_df, dimensions_df = self.load_training_data()
        
        if examples_df is not None and not examples_df.empty:
            st.header("📁 Training Data Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Entity distribution
                fig_entities = self.create_entity_distribution_chart(geometry_df)
                if fig_entities:
                    st.plotly_chart(fig_entities, use_container_width=True)
            
            with col2:
                # Layer analysis
                fig_layers = self.create_layer_analysis_chart(geometry_df)
                if fig_layers:
                    st.plotly_chart(fig_layers, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dimension analysis
                fig_dims = self.create_dimension_analysis_chart(dimensions_df)
                if fig_dims:
                    st.plotly_chart(fig_dims, use_container_width=True)
            
            with col2:
                # Training timeline
                fig_timeline = self.create_training_timeline(examples_df)
                if fig_timeline:
                    st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Model recommendations
        self.display_model_recommendations()
        
        # Detailed statistics
        st.header("📋 Detailed Statistics")
        
        # Training data breakdown
        if self.data_stats:
            st.subheader("Training Data Breakdown")
            
            data_breakdown = pd.DataFrame([
                {"Metric": "Total Training Examples", "Value": self.data_stats.get('total_examples', 0)},
                {"Metric": "Total Entities", "Value": self.data_stats.get('total_entities', 0)},
                {"Metric": "Dimensioned Entities", "Value": self.data_stats.get('total_dimensioned_entities', 0)},
                {"Metric": "Dimensioning Ratio", "Value": f"{(self.data_stats.get('total_dimensioned_entities', 0) / max(self.data_stats.get('total_entities', 1), 1) * 100):.1f}%"}
            ])
            
            st.dataframe(data_breakdown, use_container_width=True)
        
        # Model configuration
        if self.model_stats and 'training_metadata' in self.model_stats:
            metadata = self.model_stats['training_metadata']
            
            st.subheader("Model Configuration")
            
            if 'hyperparameters' in metadata:
                hyperparams = metadata['hyperparameters']
                
                config_data = []
                for key, value in hyperparams.items():
                    config_data.append({"Parameter": key, "Value": str(value)})
                
                config_df = pd.DataFrame(config_data)
                st.dataframe(config_df, use_container_width=True)
            
            # Training information
            st.subheader("Training Information")
            
            training_info = pd.DataFrame([
                {"Info": "Training Date", "Value": metadata.get('training_date', 'Unknown')},
                {"Info": "Model Version", "Value": metadata.get('model_versions', {}).get('entity_importance', 'Unknown')},
                {"Info": "Training Framework", "Value": "TensorFlow Keras with Hyperparameter Tuning"},
                {"Info": "Model Type", "Value": "Deep Neural Network with Ensemble"}
            ])
            
            st.dataframe(training_info, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🧠 **AI Dimensioning Dashboard** | "
            "Built with Streamlit, Plotly, and TensorFlow | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def main():
    """Main function to run the dashboard."""
    dashboard = AIModelDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()