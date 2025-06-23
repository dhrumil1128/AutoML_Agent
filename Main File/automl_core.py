# automl_core.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Import more models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, precision_score, recall_score, roc_auc_score
import pickle
import os
import streamlit as st
import numpy as np # Import numpy

# Function to fetch data from database - This should be outside run_automl_pipeline
# as it's a separate utility
def fetch_data_from_db(db_type, db_host, db_port, db_user, db_password, db_name, sql_query):
    # This function's implementation should be provided externally or by the user.
    # For now, it's a placeholder.
    st.error("Database connection functionality is a placeholder. Please implement `fetch_data_from_db`.")
    return None

def run_automl_pipeline(df: pd.DataFrame, target_column: str, problem_type: str):
    """
    Runs the end-to-end AutoML pipeline with expanded models and hyperparameter tuning.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        problem_type (str): "Classification" or "Regression".

    Returns:
        tuple: (best_model, evaluation_report, visualizations_data_dict)
               best_model (sklearn model object): The best trained model.
               evaluation_report (dict): Dictionary of evaluation metrics.
               visualizations_data_dict (dict): Data to generate visualizations.
    """
    st.write(f"Starting AutoML process for {problem_type} on target '{target_column}'...")

    # Separate target variable
    # For clustering, target_column might be None, handle this in X.drop
    if target_column and problem_type != "Clustering":
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else: # For clustering or if no target is specified
        X = df
        y = None # No target for clustering

    # --- Rare Class Handling for Classification Problems ---
    if problem_type == "Classification" and y is not None:
        st.write("Checking for and handling rare classes in the target column...")
        class_counts = y.value_counts()
        rare_class_threshold = 2 # At least 2 samples required for stratify to work
        rare_classes = class_counts[class_counts < rare_class_threshold].index.tolist()
        
        if rare_classes:
            st.warning(f"Identified rare classes in '{target_column}' with less than {rare_class_threshold} members: {', '.join(map(str, rare_classes))}")
            st.warning("These will be grouped into a single 'Rare_Class' category for robust training.")
            y = y.replace(rare_classes, 'Rare_Class')
            new_class_counts = y.value_counts()
            st.info(f"New target class distribution after handling rare classes:\n{new_class_counts.to_string()}")
        else:
            st.info("No rare classes detected in the target column requiring special handling.")

    # --- Data Splitting ---
    # Only split if not clustering AND target column is provided
    X_train, X_test, y_train, y_test = None, None, None, None
    if problem_type != "Clustering" and y is not None:
        st.write("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                            stratify=y if problem_type == "Classification" else None)
    else: # For clustering or no target, use full X as X_train for preprocessing
        st.write("Using full dataset for preprocessing (no train/test split for clustering or missing target).")
        X_train = X # No train/test split, use full data for training in clustering context
        # X_test and y_test will remain None, handled by later logic

    # --- Preprocessing Pipeline ---
    st.write("Setting up data preprocessing pipeline...")
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    # --- Model Selection & Hyperparameter Tuning ---
    st.write("Initializing models and setting up hyperparameter tuning...")

    models_and_params = {}

    if problem_type == "Classification":
        models_and_params = {
            'Logistic Regression': {
                # Changed solver to 'lbfgs' for multiclass and added multi_class
                'model': LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=1000), 
                'params': {'classifier__C': [0.1, 1.0, 10.0]} 
            },
            'Random Forest Classifier': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]}
            },
            'Gradient Boosting Classifier': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
            },
            'Support Vector Classifier': {
                'model': SVC(random_state=42, probability=True), # probability=True needed for ROC AUC
                'params': {'classifier__C': [0.1, 1.0], 'classifier__kernel': ['linear', 'rbf']}
            },
            'K-Nearest Neighbors Classifier': {
                'model': KNeighborsClassifier(),
                'params': {'classifier__n_neighbors': [3, 5]}
            },
            'Decision Tree Classifier': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {'classifier__max_depth': [None, 5, 10]}
            }
        }
        scoring_metric = 'accuracy'
        
    elif problem_type == "Regression":
        models_and_params = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Random Forest Regressor': {
                'model': RandomForestRegressor(random_state=42),
                'params': {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [None, 10]}
            },
            'Gradient Boosting Regressor': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.01, 0.1]}
            },
            'K-Nearest Neighbors Regressor': {
                'model': KNeighborsRegressor(),
                'params': {'regressor__n_neighbors': [3, 5]}
            },
            'Decision Tree Regressor': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {'regressor__max_depth': [None, 5, 10]}
            }
        }
        scoring_metric = 'neg_mean_squared_error'
    
    elif problem_type == "Clustering":
        # For clustering, we generally don't have a target or use GridSearchCV with scoring in the same way.
        # This section needs to be developed to include clustering algorithms (e.g., KMeans, DBSCAN).
        # For now, it will simply skip model training if clustering is selected.
        st.warning("Clustering models are not yet implemented in `automl_core.py`.")
        return None, {"model_comparisons": {}, "best_model": {"name": "Clustering (No Models)", "parameters": {}, "score": 0}}, {}


    best_model = None
    best_score_overall = -float('inf') 
    evaluation_report = {"model_comparisons": {}}
    visualizations_data_dict = {}
    best_model_name = "N/A"
    
    # Store y_test and X_test for final visualizations with the best model
    # (These are not changing during the loop, but needed for best_model evaluation)
    evaluation_report['y_test'] = y_test
    evaluation_report['X_test'] = X_test


    for name, config in models_and_params.items():
        st.write(f"\nTraining and tuning {name}...")
        
        pipeline_step_name = 'classifier' if problem_type == "Classification" else 'regressor'

        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       (pipeline_step_name, config['model'])])
        
        grid_search = GridSearchCV(full_pipeline, config['params'], cv=3, scoring=scoring_metric, n_jobs=-1)
        
        try: # Added try-except for GridSearchCV.fit
            with st.spinner(f"Tuning {name}..."):
                grid_search.fit(X_train, y_train) # Train on X_train, y_train
            
            model_tuned = grid_search.best_estimator_
            
            st.write(f"  Best parameters for {name}: {grid_search.best_params_}")
            
            # Make predictions with the best tuned model
            y_pred = model_tuned.predict(X_test) # Predict on X_test

            current_model_metrics = {}
            if problem_type == "Classification":
                score = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                current_model_metrics.update({"Accuracy": score, "F1-Score": f1, "Precision": precision, "Recall": recall})

                # Add ROC AUC only for binary classification and if model has predict_proba
                if len(y.unique()) == 2 and hasattr(model_tuned, 'predict_proba'):
                    try:
                        y_proba = model_tuned.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_proba)
                        current_model_metrics["ROC AUC"] = roc_auc
                    except Exception as e:
                        st.warning(f"Could not calculate ROC AUC for {name}: {e}")
                        current_model_metrics["ROC AUC"] = np.nan # Use NaN if calculation fails
                else:
                    current_model_metrics["ROC AUC"] = np.nan # Set NaN if not binary or no predict_proba

                st.write(f"  {name} Accuracy: {score:.4f}, F1-Score: {f1:.4f}")

            else: # Regression
                score = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"  {name} MSE: {score:.4f}, R2: {r2:.4f}")
                current_model_metrics.update({"MSE": score, "R2": r2})
            
            evaluation_report["model_comparisons"][name] = current_model_metrics


            # Check if this is the best model so far based on the primary scoring metric
            if grid_search.best_score_ > best_score_overall:
                best_score_overall = grid_search.best_score_
                best_model = model_tuned
                best_model_name = name
                st.write(f"  --> {name} is the current best model based on tuning!")

        except Exception as e:
            st.error(f"Error training or tuning {name}: {e}")
            st.info(f"Skipping {name} due to error.")
            # Do not update best_model or current_model_metrics if training fails

    # Populate best_model details and test metrics AFTER the loop, using X_test and y_test
    if best_model and y_test is not None: # Ensure best_model exists and y_test is available
        evaluation_report["best_model"] = {
            "name": best_model_name,
            # Get params from the actual model's estimator within the pipeline
            "parameters": best_model.named_steps[pipeline_step_name].get_params(), 
            "score": best_score_overall # This is the score from GridSearchCV's cross-validation
        }
        
        y_pred_best = best_model.predict(X_test)

        if problem_type == "Classification":
            accuracy_best = accuracy_score(y_test, y_pred_best)
            f1_best = f1_score(y_test, y_pred_best, average='weighted', zero_division=0)
            precision_best = precision_score(y_test, y_pred_best, average='weighted', zero_division=0)
            recall_best = recall_score(y_test, y_pred_best, average='weighted', zero_division=0)
            
            evaluation_report["best_model_test_metrics"] = {
                "Accuracy": accuracy_best,
                "F1-Score": f1_best,
                "Precision": precision_best,
                "Recall": recall_best
            }
            # Only add ROC AUC if binary and model supports predict_proba
            if len(y.unique()) == 2 and hasattr(best_model, 'predict_proba'):
                 try:
                    y_proba_best = best_model.predict_proba(X_test)[:, 1]
                    roc_auc_best = roc_auc_score(y_test, y_proba_best)
                    evaluation_report["best_model_test_metrics"]["ROC AUC"] = roc_auc_best
                 except Exception as e:
                    st.warning(f"Could not calculate ROC AUC for best model: {e}")
                    evaluation_report["best_model_test_metrics"]["ROC AUC"] = np.nan
            else:
                evaluation_report["best_model_test_metrics"]["ROC AUC"] = np.nan

            # Data for Confusion Matrix
            # Convert to list for JSON serialization if needed
            visualizations_data_dict['confusion_matrix'] = {'y_true': y_test.tolist(), 'y_pred': y_pred_best.tolist(), 'labels': y.unique().tolist()}

        elif problem_type == "Regression":
            mse_best = mean_squared_error(y_test, y_pred_best)
            r2_best = r2_score(y_test, y_pred_best)
            evaluation_report["best_model_test_metrics"] = {
                "MSE": mse_best,
                "R2": r2_best
            }
            # Data for Predicted vs. Actual plot
            # Convert to list for JSON serialization if needed
            visualizations_data_dict['predicted_vs_actual'] = {'y_true': y_test.tolist(), 'y_pred': y_pred_best.tolist()}
        
    else: # No best model found or y_test not available (e.g., clustering)
        st.warning("No best model could be trained or evaluated. Check logs for errors during model training.")
        evaluation_report["best_model"] = {"name": "No Model Trained", "parameters": {}, "score": 0}
        evaluation_report["best_model_test_metrics"] = {}

    # --- Save the Best Model ---
    model_filename = f"best_ml_model_{problem_type.lower()}.pkl"
    if best_model: # Only attempt to save if a best_model was found
        try:
            with open(model_filename, 'wb') as file:
                pickle.dump(best_model, file)
            st.success(f"Best model saved as: `{model_filename}`")
            evaluation_report['saved_model_path'] = os.path.abspath(model_filename)
        except Exception as e:
            st.error(f"Error saving the model: {e}")
            evaluation_report['saved_model_path'] = "Error saving model."
    else:
        st.info("No model to save as best model was not trained successfully.")
        evaluation_report['saved_model_path'] = "No model saved."

    st.write("AutoML process completed.")
    return best_model, evaluation_report, visualizations_data_dict