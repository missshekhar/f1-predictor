
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils import (fetch_race_results, preprocess_data, calculate_grid_position_trends,
                   calculate_driver_constructor_synergy, calculate_consistency_metrics,
                   calculate_race_incident_rates, calculate_circuit_performance,
                   calculate_qualifying_performance)

years = range(2000, 2024)

races = fetch_race_results(years)

df = pd.DataFrame(races)

df = calculate_grid_position_trends(df)
df = calculate_driver_constructor_synergy(df)
df = calculate_consistency_metrics(df)
df = calculate_race_incident_rates(df)
df = calculate_circuit_performance(df)
df = calculate_qualifying_performance(df)

df_processed, label_encoders = preprocess_data(df)

X = df_processed.drop(columns=['Position', 'Status', 'FastestLapTime', 'Incident'])
y = df_processed['Position']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [200, 300, 400],  
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}


grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

train_accuracy = best_clf.score(X_train, y_train)
test_accuracy = best_clf.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")

feature_importances = best_clf.feature_importances_
feature_names = X.columns

print("\nFeature Importances:")
for feature_name, importance in zip(feature_names, feature_importances):
    print(f"{feature_name}: {importance:.4f}")

joblib.dump(best_clf, 'f1_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\nModel and label encoders saved.")
