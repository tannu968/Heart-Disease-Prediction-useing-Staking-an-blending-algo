# Heart-Disease-Prediction-useing-Staking-an-blending-algo
This project aims to predict the presence of heart disease in patients using machine learning techniques. The dataset used is the "Heart Disease" dataset, which includes various health metrics.

## Dataset

The dataset `heart.csv` contains several features, including:
- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male; 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol
- **fbs**: Fasting blood sugar (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
- **target**: Heart disease (1 = presence; 0 = absence)

## Algorithms Used

This project utilizes the following machine learning algorithms:
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Classifier**
- **Stacking Classifier** (using Logistic Regression as the final estimator)

## Requirements

To run this project, you need the following Python packages:
- numpy
- pandas
- scikit-learn

You can install them using pip:
```bash
pip install numpy pandas scikit-learn
