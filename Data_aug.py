import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

import giskard
from giskard import Model, Dataset, testing, GiskardClient

# Constants.
COLUMN_TYPES = {
    "account_check_status": "category",
    "duration_in_month": "numeric",
    "credit_history": "category",
    "purpose": "category",
    "credit_amount": "numeric",
    "savings": "category",
    "present_employment_since": "category",
    "installment_as_income_perc": "numeric",
    "sex": "category",
    "personal_status": "category",
    "other_debtors": "category",
    "present_residence_since": "numeric",
    "property": "category",
    "age": "category",
    "other_installment_plans": "category",
    "housing": "category",
    "credits_this_bank": "numeric",
    "job": "category",
    "people_under_maintenance": "numeric",
    "telephone": "category",
    "foreign_worker": "category",
}

TARGET_COLUMN_NAME = "default"

COLUMNS_TO_SCALE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "numeric"]
COLUMNS_TO_ENCODE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "category"]

# Paths.
DATA_URL = "https://raw.githubusercontent.com/Giskard-AI/giskard-examples/main/datasets/credit_scoring_classification_model_dataset/german_credit_prepared.csv"

df = pd.read_csv(DATA_URL, keep_default_na=False, na_values=["_GSK_NA_"])
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(columns=TARGET_COLUMN_NAME), df[TARGET_COLUMN_NAME],
                                                    test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN_NAME])
raw_data = pd.concat([X_test, Y_test], axis=1)
wrapped_data = Dataset(
    df=raw_data,  # A pandas.DataFrame that contains the raw data (before all the pre-processing steps) and the actual ground truth variable (target).
    target=TARGET_COLUMN_NAME,  # Ground truth variable.
    name='German credit scoring dataset',  # Optional.
    cat_columns=COLUMNS_TO_ENCODE  # Optional, but is a MUST if available. Inferred automatically if not.
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, COLUMNS_TO_SCALE),
    ("cat", categorical_transformer, COLUMNS_TO_ENCODE),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=100))
])

pipeline.fit(X_train, Y_train)

pred_train = pipeline.predict(X_train)
pred_test = pipeline.predict(X_test)

print(classification_report(Y_test, pred_test))

wrapped_model = Model(
    model=pipeline,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type="classification",  # Either regression, classification or text_generation.
    name="Credit scoring classifier",  # Optional.
    classification_labels=pipeline.classes_.tolist(),  # Their order MUST be identical to the prediction_function's output order.
    feature_names=list(COLUMN_TYPES.keys()),  # Default: all columns of your dataset.
    # classification_threshold=0.5  # Default: 0.5.
)

# Validate wrapped model.
print(classification_report(Y_test, pipeline.classes_[wrapped_model.predict(wrapped_data).raw_prediction]))
#results = giskard.scan(wrapped_model, wrapped_data)
#print(results)
# Prediction on the entire dataset
# Split the data into train and test
Y = df['default']
X = df.drop(columns="default")
Y_pred = pipeline.predict(X)

# Calculate performance metrics for each data slice
data_slices_performance = []
for idx, row in X.iterrows():
    predicted_class = Y_pred[idx]
    true_class = Y[idx]
    is_underperforming = (predicted_class != true_class)
    data_slices_performance.append(is_underperforming)

# Identify underperforming data slices
underperforming_indices = [idx for idx, is_underperforming in enumerate(data_slices_performance) if is_underperforming]
#print("Number of Underperforming Data Slices:", underperforming_indices)

# Filter data slices
underperforming_data = X.iloc[underperforming_indices]
underperforming_labels = Y.iloc[underperforming_indices]

# Apply preprocessing to underperforming data
augmented_underperforming_data = preprocessor.transform(underperforming_data)

# Apply ADASYN or SMOTE to balance the classes on preprocessed data
adasyn = ADASYN(sampling_strategy='auto', random_state=42)
smote = SMOTE(sampling_strategy='auto', random_state=42)
augmented_underperforming_data, augmented_underperforming_labels = smote.fit_resample(augmented_underperforming_data, underperforming_labels)

# Apply preprocessing to well-performing data
well_performing_indices = [idx for idx, is_underperforming in enumerate(data_slices_performance) if not is_underperforming]
well_performing_data = X.iloc[well_performing_indices]
well_performing_labels = Y.iloc[well_performing_indices]
preprocessed_well_performing_data = preprocessor.transform(well_performing_data)
# Convert augmented data and well-performing data to DataFrames
augmented_df = pd.DataFrame(augmented_underperforming_data, columns=COLUMNS_TO_SCALE + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(input_features=COLUMNS_TO_ENCODE)))
preprocessed_well_performing_df = pd.DataFrame(preprocessed_well_performing_data, columns=COLUMNS_TO_SCALE + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(input_features=COLUMNS_TO_ENCODE)))

# Combine augmented underperforming data with preprocessed well-performing data
combined_data = pd.concat([augmented_df, preprocessed_well_performing_df])
combined_labels = pd.concat([pd.Series(augmented_underperforming_labels), well_performing_labels])

# Retrain the model on the combined dataset
clf_logistic_regression_augmented = Pipeline(steps=[
    ('classifier', LogisticRegression(max_iter=1000))
])

X_prime = combined_data
Y_prime = combined_labels

X_train_prime,X_test_prime,Y_train_prime,Y_test_prime = train_test_split(X_prime,Y_prime,test_size=0.20, random_state=30, stratify=Y_prime)

clf_logistic_regression_augmented.fit(combined_data,combined_labels)

# Evaluate the retrained model on the test set
score_retrained = clf_logistic_regression_augmented.score(preprocessor.transform(X_test), Y_test)
print("Retrained Model Score:", score_retrained)
pred_test = clf_logistic_regression_augmented.predict(X_test_prime)
print(classification_report(Y_test_prime, pred_test))






