# LendingClub Loan Default Prediction

A comprehensive machine learning project that predicts loan defaults using LendingClub historical data through deep neural networks.

## 📊 Project Overview

This project implements a **6-phase systematic approach** to predict loan defaults using advanced machine learning techniques:

1. **Phase 1**: Exploratory Data Analysis (EDA) and Data Visualization
2. **Phase 2**: Data Preprocessing and Feature Engineering
3. **Phase 3**: Neural Network Architecture Design
4. **Phase 4**: Model Training and Hyperparameter Tuning
5. **Phase 5**: Model Evaluation and Performance Analysis
6. **Phase 6**: Results Interpretation and Business Recommendations

## 🎯 Key Results

- **Accuracy**: 86.41%
- **Precision**: 85.9%
- **Recall**: 99.4%
- **ROC-AUC**: 86.0%
- **F1-Score**: 92.16%
- **Estimated Loss Prevention**: $72.4 Million

## 📈 Dataset Information

- **Total Records**: 396,030 loan applications
- **Features**: 27 variables including loan amount, interest rate, employment history, credit history, etc.
- **Target Variable**: Binary classification (Fully Paid vs Charged Off)
- **Class Distribution**: 80.4% Fully Paid, 19.6% Charged Off

### Key Features
- **Loan Amount**: $500 - $40,000 (Mean: $14,114)
- **Interest Rate**: Primary predictor of default risk
- **Loan Term**: 36 or 60 months
- **Employment Length**: 0-10+ years
- **Annual Income**: Self-reported borrower income
- **Credit History**: Length of credit history, revolving utilization
- **Geographic Data**: ZIP code patterns

## 🧠 Model Architecture

**Deep Neural Network** with the following specifications:
- **Layers**: 12 layers total
- **Parameters**: 23,809 trainable parameters
- **Architecture**: Dense layers with Dropout and BatchNormalization
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary crossentropy with class weights

### Model Features
- **Early Stopping**: Monitors validation recall
- **Learning Rate Reduction**: Dynamic LR adjustment
- **Class Weighting**: Enhanced weights for minority class (defaults)
- **Batch Normalization**: Improved training stability
- **Dropout**: Prevents overfitting

## 🔧 Data Preprocessing

### Feature Engineering
- **Missing Value Handling**: Strategic imputation based on feature type
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Numerical Scaling**: StandardScaler for numerical features
- **Date Processing**: Extracted year and month from date fields
- **ZIP Code Processing**: Extracted first 3 digits for geographic patterns
- **Employment Length**: Converted to numerical format

### Feature Selection
- **Correlation Analysis**: Identified highly correlated features
- **Feature Importance**: Ranked features by predictive power
- **Dimensionality**: Processed 100+ features after encoding

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 86.41% | Overall prediction accuracy |
| Precision | 85.9% | Precision for default predictions |
| Recall | 99.4% | Ability to catch actual defaults |
| F1-Score | 92.16% | Balanced precision-recall metric |
| ROC-AUC | 86.0% | Area under ROC curve |
| Training Time | 1.57 minutes | Model training duration |

## 💼 Business Value

### Risk Assessment
- **Default Identification Rate**: 33.0% of actual defaults correctly identified
- **Loss Prevention**: Estimated $72.4 million in prevented losses
- **Model Improvement**: +7.49% improvement over baseline accuracy

### Key Insights
- **Interest Rate**: Primary indicator of default risk (negative correlation)
- **Loan Term**: 60-month loans show higher default rates
- **Geographic Patterns**: Certain ZIP codes show higher default rates
- **Credit Utilization**: Higher revolving utilization increases default risk
- **Employment Stability**: Longer employment history reduces default risk

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencies
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- scikit-learn>=1.3.0
- tensorflow>=2.18.0
- jupyter>=1.0.0
- plotly>=5.0.0

### Installation
```bash
git clone https://github.com/ruthwikchikoti/LendingClub_Loan_Prediction.git
cd LendingClub_Loan_Prediction
pip install -r requirements.txt
```

### Usage
```bash
jupyter notebook LendingClub_Loan_Default_Prediction.ipynb
```

## 📁 Project Structure

```
LendingClub_Loan_Prediction/
├── LendingClub_Loan_Default_Prediction.ipynb    # Main analysis notebook
├── lending_club_loan_default_model.h5           # Trained model
├── preprocessor.pkl                             # Data preprocessor
├── lending_club_info.csv                       # Feature descriptions
├── lending_club_loan_two.csv                   # Main dataset
├── model_architecture.png                      # Model visualization
├── requirements.txt                             # Dependencies
└── README.md                                   # Project documentation
```

## 🔍 Key Findings

### Top Predictive Features
1. **Interest Rate**: Strongest predictor of default risk
2. **Loan Term**: 60-month loans higher risk than 36-month
3. **Revolving Utilization**: High utilization indicates financial stress
4. **Geographic Location**: ZIP code patterns correlate with defaults
5. **Credit History**: Longer credit history reduces default risk

### Model Performance Analysis
- **High Recall (99.4%)**: Excellent at identifying potential defaults
- **Good Precision (85.9%)**: Minimizes false positives
- **Balanced F1-Score (92.16%)**: Strong overall performance
- **Generalization**: Minimal overfitting with robust validation

## 🎯 Future Enhancements

1. **Advanced Ensemble Methods**: Combine multiple models
2. **Feature Engineering**: Additional derived features
3. **Real-time Scoring**: API deployment for live predictions
4. **Explainable AI**: SHAP values for feature importance
5. **Alternative Data**: Integration of non-traditional data sources

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Distribution Analysis**: Loan amounts, interest rates, terms
- **Correlation Heatmaps**: Feature relationships
- **Geographic Analysis**: Default patterns by location
- **Model Performance**: ROC curves, precision-recall curves
- **Feature Importance**: Top predictive features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

**Ruthwik Chikoti**
- GitHub: [@ruthwikchikoti](https://github.com/ruthwikchikoti)
- Project Link: [LendingClub Loan Prediction](https://github.com/ruthwikchikoti/LendingClub_Loan_Prediction)

---

*This project demonstrates end-to-end machine learning workflow for financial risk assessment, showcasing data science techniques from EDA to model deployment.*
