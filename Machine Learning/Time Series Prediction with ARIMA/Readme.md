
## Feature Selection

The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features. The example here uses the chi squared (chi^2) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset.

Below are the columns of the dataset:

Pregnancies - Number of times pregnant
Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure - Diastolic blood pressure (mm Hg)
SkinThickness - Triceps skin fold thickness (mm)
Insulin - 2-Hour serum insulin (mu U/ml)
BMI - Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction - Diabetes pedigree function
Age - Age (years)

Also experimented with 'Recursive Feature Selection' to identify the attributes which contribute the most for model building.
Checked the feature importance by 'Extra Tree Ensemble' and 'Principal Component Analysis'
