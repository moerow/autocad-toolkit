# AI Model Performance Dashboard 🧠

A comprehensive, interactive dashboard for monitoring your AI dimensioning model's health, accuracy, and performance.

## Features

### 🏥 **Model Health Overview**
- **Real-time model status** (Excellent/Good/Fair/Poor)
- **Key metrics** at a glance
- **Training data statistics**
- **Model performance indicators**

### 📈 **Performance Analytics**
- **Accuracy gauges** with color-coded thresholds
- **Ensemble vs individual model performance**
- **Precision, Recall, F1-Score metrics**
- **Validation accuracy tracking**

### 📊 **Training Progress Visualization**
- **Training/validation curves** over epochs
- **Loss function progression**
- **Learning rate tracking**
- **Overfitting detection**

### 🎯 **Classification Analysis**
- **Confusion matrix** for entity importance
- **Class-wise performance breakdown**
- **Prediction confidence analysis**

### 📁 **Training Data Insights**
- **Entity type distribution** across training files
- **Layer analysis** showing most common layers
- **Dimension type breakdown**
- **Training timeline** showing data collection progress

### 🎯 **Intelligent Recommendations**
- **Automated model health assessment**
- **Performance improvement suggestions**
- **Training data recommendations**
- **Model maintenance alerts**

## Installation

```bash
# Install dashboard dependencies
pip install -r requirements_dashboard.txt

# Or install individually
pip install streamlit plotly pandas numpy scikit-learn tensorflow keras-tuner
```

## Quick Start

### Method 1: Auto-Launch Script
```bash
# Simple one-click launch
python run_dashboard.py
```

### Method 2: Direct Streamlit
```bash
# Launch with Streamlit directly
streamlit run ai_dashboard.py
```

### Method 3: Custom Storage Path
```bash
# Launch with custom AI training data path
streamlit run ai_dashboard.py -- --storage-path "C:/My_AI_Training_Data"
```

## Dashboard Sections

### 1. **Model Health Overview**
```
🏥 Model Health Overview
├── Model Status: Excellent ✅
├── Training Examples: 147
├── Total Entities: 35,642
└── Dimensioned Entities: 12,847
```

### 2. **Performance Metrics**
- **Accuracy Gauge**: Visual gauge showing model accuracy (0-100%)
- **Ensemble Accuracy**: Performance of ensemble voting
- **Validation Accuracy**: Generalization performance
- **Detailed Metrics**: Precision, Recall, F1-Score, Validation Loss

### 3. **Training Progress**
- **Accuracy Curves**: Training vs validation accuracy over epochs
- **Loss Curves**: Training vs validation loss progression
- **Learning Rate**: Adaptive learning rate changes
- **F1 Score**: Model performance evolution

### 4. **Classification Analysis**
- **Confusion Matrix**: Entity importance classification accuracy
- **Class Performance**: Per-class precision and recall
- **Prediction Confidence**: Model certainty in predictions

### 5. **Training Data Analysis**
- **Entity Distribution**: Pie chart of entity types in training data
- **Layer Analysis**: Bar chart of most common layers
- **Dimension Types**: Distribution of dimension types
- **Training Timeline**: When training data was collected

### 6. **Recommendations**
- **Performance Alerts**: Automatic detection of low accuracy
- **Data Recommendations**: Suggestions for improving training data
- **Model Maintenance**: Alerts for model age and retraining needs

## Dashboard Screenshots

### Model Health Overview
```
🏥 Model Health Overview
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Model Status  │ Training Examples│   Total Entities│Dimensioned Entities│
│   Excellent ✅  │       147       │     35,642      │     12,847      │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Performance Gauges
```
📈 Performance Metrics
┌─────────────────┬─────────────────┬─────────────────┐
│  Model Accuracy │Ensemble Accuracy│Validation Accuracy│
│      [96%]      │      [98%]      │      [94%]      │
│   ████████████  │   ████████████  │   ████████████  │
└─────────────────┴─────────────────┴─────────────────┘
```

### Training Progress
```
📊 Training Progress
┌─────────────────────────────────────────────────────────────────────┐
│                    Training/Validation Accuracy                     │
│  1.0 ┤                                                         ╭─   │
│  0.9 ┤                                                   ╭─────╯    │
│  0.8 ┤                                             ╭─────╯          │
│  0.7 ┤                                       ╭─────╯                │
│  0.6 ┤                                 ╭─────╯                      │
│  0.5 ┤                           ╭─────╯                            │
│  0.4 ┤                     ╭─────╯                                  │
│  0.3 ┤               ╭─────╯                                        │
│  0.2 ┤         ╭─────╯                                              │
│  0.1 ┤   ╭─────╯                                                    │
│  0.0 ┤───╯                                                          │
│      └─────────────────────────────────────────────────────────────│
│           1    10    20    30    40    50    60    70    80    90   │
│                                 Epochs                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Health Indicators

### 🟢 **Excellent (95%+ Accuracy)**
- Model is performing optimally
- Ready for production use
- No immediate action needed

### 🟡 **Good (90-95% Accuracy)**
- Model is performing well
- Consider minor fine-tuning
- Monitor for degradation

### 🟠 **Fair (80-90% Accuracy)**
- Model needs improvement
- Add more training data
- Consider hyperparameter tuning

### 🔴 **Poor (<80% Accuracy)**
- Model requires immediate attention
- Significant training data needed
- Consider model architecture changes

## Recommendations Engine

The dashboard provides intelligent recommendations:

### **Performance Recommendations**
```
🎯 Model Improvement Recommendations

✅ Excellent Accuracy: Ensemble accuracy is 98.2%. Model is performing 
   excellently and ready for production use.

ℹ️ Model Age: Model was trained 45 days ago. Consider retraining with 
   recent project data for optimal performance.

⚠️ Limited Training Data: 89 training examples. Adding more diverse DWG 
   files will improve model robustness.
```

### **Data Quality Recommendations**
- **Entity Distribution**: Suggests balancing entity types
- **Layer Coverage**: Recommends adding missing layer types
- **Dimension Variety**: Suggests including different dimension types

### **Training Recommendations**
- **Retraining Schedule**: Suggests optimal retraining frequency
- **Hyperparameter Tuning**: Recommends parameter adjustments
- **Data Augmentation**: Suggests additional training data sources

## Accessing the Dashboard

### **Local Access**
```
🚀 Dashboard URL: http://localhost:8501
🔄 Auto-refresh: Dashboard updates in real-time
📱 Mobile-friendly: Responsive design for all devices
```

### **Network Access**
```bash
# Allow network access
streamlit run ai_dashboard.py --server.address=0.0.0.0 --server.port=8501
```

## Troubleshooting

### **Common Issues**

1. **Dashboard Won't Start**
   ```bash
   # Check Streamlit installation
   pip install streamlit
   
   # Check for port conflicts
   netstat -an | grep 8501
   ```

2. **No Data Displayed**
   - Ensure AI models are trained
   - Check storage path configuration
   - Verify database exists

3. **Performance Issues**
   - Close other resource-intensive applications
   - Check available RAM
   - Consider reducing data visualization complexity

### **Error Messages**

**"No Model Found"**
- Train your AI models first using `train_ai_models.py`
- Check storage path in sidebar

**"Database Connection Failed"**
- Ensure training data database exists
- Check file permissions
- Verify storage path

**"Failed to Load Statistics"**
- Retrain models to generate fresh statistics
- Check for corrupted model files

## Dashboard Updates

### **Auto-Refresh**
- Dashboard automatically updates when new training data is added
- Model statistics refresh when models are retrained
- Use "🔄 Refresh Data" button for manual updates

### **Version Tracking**
- Dashboard shows model version and training date
- Tracks model performance over time
- Provides model comparison features

## Integration with Training

### **Seamless Integration**
```python
# Train models and immediately view results
python train_ai_models.py --dwg-dir "C:/DWG_Files"
python run_dashboard.py  # View training results
```

### **Continuous Monitoring**
- Dashboard updates automatically after training
- Performance tracking over multiple training sessions
- Historical performance comparison

## Advanced Features

### **Custom Metrics**
- Add custom performance metrics
- Create custom visualizations
- Export performance reports

### **Model Comparison**
- Compare different model versions
- Track performance improvements over time
- A/B testing for model parameters

### **Export Capabilities**
- Export performance reports to PDF
- Save visualizations as images
- Export training statistics to CSV

---

## Quick Commands

```bash
# Launch dashboard
python run_dashboard.py

# Train models and view results
python train_ai_models.py --dwg-dir "C:/DWG_Files" && python run_dashboard.py

# View existing model statistics
python train_ai_models.py --stats

# Launch with custom path
streamlit run ai_dashboard.py -- --storage-path "C:/Custom_AI_Data"
```

🧠 **Monitor your AI model's health and performance with this comprehensive dashboard!**