import streamlit as st
import pandas as pd
from automl_core import run_automl_pipeline, fetch_data_from_db
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: var(--text-color); /* Use Streamlit's text color variable */
    }

    /* Main container styling */
    .css-18e3th9 { /* Main content area, this class might vary slightly between Streamlit versions */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .stApp {
        background-color: var(--background-color); /* Use Streamlit's main background color variable */
    }

    /* Header styling (Keep specific colors for branding if desired, but ensure contrast) */
    h1 {
        color: #1a237e; /* Dark blue for main title - often visible in dark mode */
        text-align: center;
        margin-bottom: 1rem;
        font-size: 2.5rem;
    }
    h2, h3, h4 {
        color: #3f51b5; /* Medium blue for section headers - often visible in dark mode */
    }

    /* Markdown text styling */
    .stMarkdown {
        line-height: 1.6;
        color: var(--text-color); /* Use Streamlit's text color variable */
    }

    /* Buttons styling */
    .stButton>button {
        background-color: var(--primary-color); /* Use Streamlit's primary accent color */
        color: white; /* White text usually works on primary accent color */
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--primary-color); /* Or a slightly darker shade if you want a hover effect */
        filter: brightness(90%); /* A subtle brightness change for hover */
    }

    /* File uploader styling */
    .stFileUploader {
        background-color: var(--secondary-background-color); /* Use Streamlit's secondary background for components */
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Shadow might need adjustment in dark mode, but often still works */
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: bold;
        color: var(--text-color); /* Use Streamlit's text color for headers */
    }

    /* Success/Error/Info messages */
    .stAlert {
        border-radius: 0.5rem;
        /* Streamlit handles alert colors automatically, no need to override background/text */
    }

    /* Sidebar styling: Use Streamlit's secondary background and border colors */
    .sidebar .sidebar-content {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-right: 1px solid var(--border-color); /* Use Streamlit's border color */
        overflow-y: hidden; /* This attempts to remove the scrollbar for the About section */
    }

    /* Selectbox styling: Use Streamlit's secondary background for component backgrounds */
    div[data-testid="stSelectbox"] > div:first-child {
        background-color: var(--secondary-background-color); /* Use secondary background */
        border-radius: 0.5rem; /* Match other rounded elements */
        color: var(--text-color); /* Ensure text inside is readable */
    }
    
    /* Ensure the actual selected text in the selectbox is readable */
    div[data-testid="stSelectbox"] .st-cg { /* Targeting a common inner class for the display text */
        color: var(--text-color);
    }

    /* Footer Styling: Use Streamlit's secondary background, text, and border colors */
    .footer {
        font-size: 0.85rem;
        color: var(--text-color); /* Use Streamlit's text color variable */
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem; /* Add some space above the footer */
        border-top: 1px solid var(--border-color); /* Use Streamlit's border color */
        background-color: var(--secondary-background-color); /* Use secondary background */
        width: 100%;
    }
    .footer a {
        color: var(--primary-color); /* Use primary color for links in footer */
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# --- Header & Introduction ---
st.title("🤖 Intelligent AutoML Agent")
st.markdown("""
Welcome to your personal AI-powered data scientist!
Upload your CSV, tell me what to predict, and I'll handle the rest – from data cleaning to building and explaining the best machine learning model.
""")

st.divider()

# -------------------------------------------------------------------- Step 1: Data Ingestion -----------------------------------------------------------
st.header("Step 1: Ingest Your Dataset")

with st.container(border=True):
    data_source_option = st.radio(
        "Choose your data source:",
        ("Upload File (CSV/JSON)", "Connect to Database (SQL Query)"),
        key="data_source_radio",
        horizontal=True
    )

    df = None # Initialize df outside the if blocks

    if data_source_option == "Upload File (CSV/JSON)":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "json"], key="file_uploader")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    st.success("CSV file uploaded successfully!")
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                    st.success("JSON file uploaded successfully!")
                else:
                    st.warning("Unsupported file type uploaded. Please use CSV or JSON.")
                    df = None # Reset df if type is unsupported

                if df is not None:
                    st.subheader("Data Preview:")
                    st.write(df.head())
                    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                    st.session_state['df'] = df # Store df in session state
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Please ensure your file is a valid CSV or JSON.")
                df = None
        else:
            st.info("Please upload your dataset to get started.")

    elif data_source_option == "Connect to Database (SQL Query)":
        st.subheader("Database Connection Details")
        col1, col2 = st.columns(2)
        with col1:
            db_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite", "SQL Server"], key="db_type")
            db_host = st.text_input("Host:", value="localhost", key="db_host")
            db_user = st.text_input("User:", key="db_user")
        with col2:
            db_port = st.text_input("Port:", value="5432" if st.session_state.get("db_type") == "PostgreSQL" else "3306", key="db_port")
            db_password = st.text_input("Password:", type="password", key="db_password")
            db_name = st.text_input("Database Name:", key="db_name")
        
        sql_query = st.text_area("SQL Query:", height=150, key="sql_query", 
                                 placeholder="e.g., SELECT * FROM your_table WHERE condition;")
        
        if st.button("Fetch Data from Database", key="fetch_db_button"):
            if not sql_query:
                st.warning("Please enter an SQL query.")
            else:
                with st.spinner("Connecting to database and fetching data..."):
                    try:
                        df = fetch_data_from_db(
                            db_type, 
                            db_host, 
                            db_port,
                            db_user, 
                            db_password, 
                            db_name, 
                            sql_query
                        )
                        if df is not None and not df.empty:
                            st.session_state['df'] = df
                            st.success("Data fetched successfully from database!")
                            st.subheader("Data Preview:")
                            st.write(df.head())
                            st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                        elif df is not None and df.empty:
                            st.warning("Query returned an empty dataset.")
                            st.session_state['df'] = None
                        else:
                            st.error("Failed to fetch data from the database. Check logs above for details.")
                            st.session_state['df'] = None
                    except Exception as e:
                        st.error(f"An unexpected error occurred during database fetch: {e}")
                        st.session_state['df'] = None

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
    else:
        df = None


# -------------------------------------------------------------------- Step 2: Configure AutoML (with auto-suggestion) ----------------------------------------------------
st.header("Step 2: Configure AutoML")

if df is not None:
    with st.container(border=True):
        all_columns = df.columns.tolist()

        # Target column selection
        # Initialize index for selectbox
        initial_target_column_index = 0
        if 'target_column' in st.session_state and st.session_state['target_column'] in all_columns:
            initial_target_column_index = all_columns.index(st.session_state['target_column'])
        
        target_column = st.selectbox(
            "Select the target column (what you want to predict):",
            ['-- Select a column --'] + all_columns, # Changed '' to explicit placeholder text
            index=initial_target_column_index + 1 if 'target_column' in st.session_state and st.session_state['target_column'] in all_columns else 0,
            key="target_column_select"
        )
        st.session_state['target_column'] = target_column if target_column else None


        # --- Auto-suggest Problem Type ---
        default_problem_type_index = 0 # Default to Classification
        problem_type_options = ("Classification", "Regression", "Clustering")

        if target_column and target_column in df.columns:
            unique_values_target = df[target_column].nunique()
            total_rows_target = df.shape[0]

            if pd.api.types.is_numeric_dtype(df[target_column]):
                # Heuristic for numerical classification vs regression
                if unique_values_target / total_rows_target < 0.1 and unique_values_target < 50:
                    inferred_problem_type = "Classification"
                    st.info(f"Target column '{target_column}' is numerical with few unique values. Suggested problem type: **Classification**.")
                else:
                    inferred_problem_type = "Regression"
                    st.info(f"Target column '{target_column}' is numerical with many unique values. Suggested problem type: **Regression**.")
            else: # Categorical (object, bool, category dtype)
                inferred_problem_type = "Classification"
                st.info(f"Target column '{target_column}' is categorical. Suggested problem type: **Classification**.")
            
            default_problem_type_index = problem_type_options.index(inferred_problem_type)

        else: # No target column selected
            st.info("No target column selected. Consider **Clustering** to find patterns in your data, or select a target for Classification/Regression.")
            default_problem_type_index = problem_type_options.index("Clustering")


        # Allow user to override auto-selection
        problem_type = st.radio(
            "Confirm or change the problem type:",
            problem_type_options,
            index=default_problem_type_index,
            key="problem_type_radio",
            horizontal=True
        )
        st.session_state['problem_type'] = problem_type
        
        if problem_type == "Clustering" and st.session_state['target_column']:
             st.warning("For Clustering, the selected target column will be ignored during model training, but can be used for analysis.")
        elif problem_type != "Clustering" and not st.session_state['target_column']:
            st.error("Please select a target column for Classification or Regression.")


        if st.button("Run AutoML", key="run_automl_button_main"):
            if 'df' not in st.session_state or st.session_state['df'] is None:
                st.error("Please upload or fetch a dataset first.")
            elif st.session_state['problem_type'] != "Clustering" and not st.session_state['target_column']:
                st.error("Please select a target column for Classification or Regression.")
            else:
                st.info(f"Running AutoML for {st.session_state['problem_type']}...")
                
                with st.spinner("Processing data, training models, and evaluating... This might take a moment!"):
                    best_model, evaluation_report, visualizations_data_dict = run_automl_pipeline(
                        st.session_state['df'],
                        st.session_state['target_column'], # Pass target_column, automl_core will handle None for clustering
                        st.session_state['problem_type']
                    )
                    
                    st.session_state['best_model'] = best_model
                    st.session_state['evaluation_report'] = evaluation_report
                    st.session_state['viz_data'] = visualizations_data_dict
                    st.session_state['automl_ran'] = True
                
                st.success("AutoML process completed!")
                st.rerun() # Rerun to display results immediately
else:
    st.info("Please upload your dataset or connect to a database to proceed with AutoML configuration.")

# -------------------------------------------------------------------- Step 3: AutoML Results & Model Download ----------------------------------------------------------
st.header("Step 3: AutoML Results & Model Download")

if st.session_state.get('automl_ran'):
    evaluation_report = st.session_state['evaluation_report']
    best_model = st.session_state['best_model']
    viz_data = st.session_state['viz_data']
    problem_type = st.session_state.get('problem_type')
    
    st.subheader("Best Model Summary") # Changed from 'Best Model: N/A' to a subheader for summary
    
    # Display best model details (Name, Cross-Validation Score, Hyperparameters)
    if evaluation_report.get("best_model"):
        st.write(f"**Model Name:** `{evaluation_report['best_model']['name']}`")
        st.write(f"**Cross-Validation Score:** `{evaluation_report['best_model']['score']:.4f}`")
        st.write(f"**Best Hyperparameters:**")
        st.json(evaluation_report['best_model']['parameters'])

        # --- New: Explanation for Best Model Choice ---
        st.markdown("---")
        st.subheader("Why this is the Best Model?")
        if problem_type == "Classification":
            st.info(
                f"The best model was chosen primarily based on its **Accuracy** and **F1-Score** during "
                f"cross-validation. Accuracy indicates the overall correct predictions, while F1-Score "
                f"provides a balance between precision and recall, especially important for imbalanced datasets."
            )
        elif problem_type == "Regression":
            st.info(
                f"For regression tasks, the best model is selected based on its **R-squared (R2) score** "
                f"and **Mean Squared Error (MSE)** during cross-validation. R2 represents the proportion of "
                f"variance in the dependent variable that can be predicted from the independent variables, "
                f"while MSE measures the average squared difference between the estimated values and the actual value."
            )
        elif problem_type == "Clustering":
            st.info(
                f"For clustering, there's no 'best' model in the supervised sense. The chosen model "
                f"({evaluation_report['best_model']['name']}) was selected as a representative clustering "
                f"algorithm for its ability to identify natural groupings in the data. "
                f"Evaluation of clustering models often involves metrics like Silhouette Score (which aims to maximize)."
            )
        st.markdown("---")
            
    # Display best model metrics
    if evaluation_report.get("best_model_test_metrics"):
        st.write("#### Performance Metrics:")
        test_metrics = evaluation_report["best_model_test_metrics"]
        # Display metrics in columns
        cols = st.columns(len(test_metrics))
        for i, (metric_name, value) in enumerate(test_metrics.items()):
            # Handle specific metrics like ROC AUC which might not be numeric for all cases
            if isinstance(value, (int, float)):
                cols[i].metric(label=metric_name, value=f"{value:.4f}")
            else:
                cols[i].metric(label=metric_name, value=str(value))

    # Model Comparison Table - ENHANCED STYLING
    if evaluation_report.get("model_comparisons"): # Check if model_comparisons exists
        st.write("#### Model Comparison:")
        # FIX: Transpose the DataFrame so models are rows and metrics are columns
        comparison_df = pd.DataFrame(evaluation_report["model_comparisons"]).T

        # Define columns for highlighting
        highlight_max_cols = []
        highlight_min_cols = []

        if problem_type == "Classification":
            # Ensure these columns exist before adding them to highlight lists
            available_cols = comparison_df.columns.tolist()
            if 'Accuracy' in available_cols: highlight_max_cols.append('Accuracy')
            if 'F1-Score' in available_cols: highlight_max_cols.append('F1-Score')
            if 'Precision' in available_cols: highlight_max_cols.append('Precision')
            if 'Recall' in available_cols: highlight_max_cols.append('Recall')
            if 'ROC AUC' in available_cols: highlight_max_cols.append('ROC AUC')
        elif problem_type == "Regression":
            available_cols = comparison_df.columns.tolist()
            if 'MSE' in available_cols: highlight_min_cols.append('MSE')
            if 'R2' in available_cols: highlight_max_cols.append('R2')
        # Clustering doesn't have these comparison metrics in this table, so no highlighting needed

        # Apply highlighting
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=highlight_max_cols, color='#B9F2FF') # Light blue for max
                         .highlight_min(axis=0, subset=highlight_min_cols, color='#FFC0CB'), # Light pink for min
            use_container_width=True
        )
    else:
        st.info("No model comparison data available yet.")


    # --- Key Visualizations ---
    st.write("#### Key Visualizations:")
    if viz_data:
        # --- Classification Visualizations ---
        if problem_type == "Classification":
            st.markdown("##### Classification Plots")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix (Best Model)**")
                if 'confusion_matrix' in viz_data:
                    y_true = viz_data['confusion_matrix']['y_true']
                    y_pred = viz_data['confusion_matrix']['y_pred']
                    labels = viz_data['confusion_matrix']['labels']
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    fig, ax = plt.subplots(figsize=(5, 4.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    ax.set_title('Confusion Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Confusion matrix data not available.")
            with col2:
                st.markdown("**Feature Importance (Best Tree-Based Model)**")
                best_model_pipeline = best_model # Directly use best_model from session_state
                if best_model_pipeline:
                    final_model = None
                    # Try to get the final estimator from a pipeline
                    if hasattr(best_model_pipeline, 'named_steps') and 'classifier' in best_model_pipeline.named_steps:
                        final_model = best_model_pipeline.named_steps['classifier']
                    elif hasattr(best_model_pipeline, 'named_steps') and 'regressor' in best_model_pipeline.named_steps:
                        final_model = best_model_pipeline.named_steps['regressor']
                    elif hasattr(best_model_pipeline, 'named_steps') and 'clusterer' in best_model_pipeline.named_steps:
                        final_model = best_model_pipeline.named_steps['clusterer']
                    else: # If it's not a pipeline, assume it's the model itself
                        final_model = best_model_pipeline
                    
                    if hasattr(final_model, 'feature_importances_'):
                        feature_importances = final_model.feature_importances_
                        try:
                            feature_names = None
                            if hasattr(best_model_pipeline, 'named_steps') and 'preprocessor' in best_model_pipeline.named_steps:
                                preprocessor = best_model_pipeline.named_steps['preprocessor']
                                if hasattr(preprocessor, 'get_feature_names_out'):
                                    try:
                                        # Convert to list to avoid numpy boolean ambiguity
                                        feature_names = preprocessor.get_feature_names_out().tolist() 
                                    except Exception:
                                        st.warning("Could not get processed feature names from preprocessor. Using generic names for feature importance plot.")
                                        feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                                else:
                                    feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                            
                            if feature_names is None: # Fallback if no preprocessor or other issues
                                if 'df' in st.session_state and st.session_state['df'] is not None and st.session_state['target_column']:
                                    original_features = st.session_state['df'].drop(columns=[st.session_state['target_column']]).columns.tolist()
                                    if len(original_features) == len(feature_importances):
                                        feature_names = original_features
                                    else:
                                        feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                                else:
                                    feature_names = [f"Feature {i}" for i in range(len(feature_importances))]

                            # FIX: Robust check for feature_names to avoid ambiguity error
                            if feature_names is not None and len(feature_names) > 0 and len(feature_importances) == len(feature_names):
                                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
                                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax, palette='viridis')
                                ax.set_title('Top 10 Feature Importances')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("Could not process feature importances into a meaningful plot (feature names mismatch or not available).")

                        except Exception as e:
                            st.error(f"Error processing feature importances: {e}")
                    else:
                        st.info("Best model does not directly provide feature importances (e.g., Linear Regression, SVM, KNN, K-Means).")
                else:
                    st.info("Best model not available to compute feature importances.")

        # --- Regression Visualizations ---
        elif problem_type == "Regression":
            st.markdown("##### Regression Plots")
            if 'predicted_vs_actual' in viz_data:
                y_true = viz_data['predicted_vs_actual']['y_true']
                y_pred = viz_data['predicted_vs_actual']['y_pred']

                fig, ax = plt.subplots(figsize=(6, 5))
                sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.6)
                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2) # Diagonal line
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Predicted vs. Actual Values')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Predicted vs. Actual plot data not available.")
            
            # Add feature importance for regression if applicable
            st.markdown("**Feature Importance (Best Tree-Based Model)**")
            best_model_pipeline = best_model
            if best_model_pipeline:
                final_model = None
                # Try to get the final estimator from a pipeline
                if hasattr(best_model_pipeline, 'named_steps') and 'regressor' in best_model_pipeline.named_steps:
                    final_model = best_model_pipeline.named_steps['regressor']
                elif hasattr(best_model_pipeline, 'named_steps') and 'classifier' in best_model_pipeline.named_steps:
                    final_model = best_model_pipeline.named_steps['classifier']
                else: # If it's not a pipeline, assume it's the model itself
                    final_model = best_model_pipeline

                if hasattr(final_model, 'feature_importances_'):
                    feature_importances = final_model.feature_importances_
                    try:
                        feature_names = None
                        if hasattr(best_model_pipeline, 'named_steps') and 'preprocessor' in best_model_pipeline.named_steps:
                            preprocessor = best_model_pipeline.named_steps['preprocessor']
                            if hasattr(preprocessor, 'get_feature_names_out'):
                                try:
                                    # Convert to list to avoid numpy boolean ambiguity
                                    feature_names = preprocessor.get_feature_names_out().tolist()
                                except Exception:
                                    st.warning("Could not get processed feature names from preprocessor. Using generic names for feature importance plot.")
                                    feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                            else:
                                feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                        
                        if feature_names is None:
                            if 'df' in st.session_state and st.session_state['df'] is not None and st.session_state['target_column']:
                                original_features = st.session_state['df'].drop(columns=[st.session_state['target_column']]).columns.tolist()
                                if len(original_features) == len(feature_importances):
                                    feature_names = original_features
                                else:
                                    feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
                            else:
                                feature_names = [f"Feature {i}" for i in range(len(feature_importances))]

                        # FIX: Robust check for feature_names to avoid ambiguity error
                        if feature_names is not None and len(feature_names) > 0 and len(feature_importances) == len(feature_names):
                            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
                            importance_df = importance_df.sort_values(by='Importance', ascending=False)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax, palette='viridis')
                            ax.set_title('Top 10 Feature Importances')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("Could not process feature importances into a meaningful plot (feature names mismatch or not available).")
                    except Exception as e:
                        st.error(f"Error processing feature importances: {e}")
                else:
                    st.info("Best model does not directly provide feature importances (e.g., Linear Regression, SVM, KNN).")
            else:
                st.info("Best model not available to compute feature importances.")
                
        # --- Clustering Visualizations ---
        elif problem_type == "Clustering":
            st.markdown("##### Clustering Visualizations")
            # You would add clustering specific visualizations here, e.g.,
            # PCA/TSNE for 2D/3D scatter plots of clusters, silhouette scores, etc.
            if 'cluster_labels' in viz_data and 'pca_components' in viz_data:
                st.info("Generating PCA plot for clusters. This might take a moment if the dataset is large.")
                try:
                    cluster_labels = viz_data['cluster_labels']
                    pca_components = viz_data['pca_components'] # Expecting 2 or 3 components

                    if len(pca_components) > 0 and pca_components[0] is not None and len(pca_components[0]) >= 2:
                        fig = plt.figure(figsize=(8, 6))
                        # Convert list of lists (from JSON) back to numpy array for plotting
                        pca_components_np = np.array(pca_components) 

                        if pca_components_np.shape[1] == 2:
                            ax = fig.add_subplot(111)
                            sns.scatterplot(x=pca_components_np[:, 0], y=pca_components_np[:, 1], 
                                            hue=cluster_labels, palette='viridis', legend='full', ax=ax)
                            ax.set_xlabel('Principal Component 1')
                            ax.set_ylabel('Principal Component 2')
                            ax.set_title('Clusters in 2D PCA Space')
                        elif pca_components_np.shape[1] >= 3:
                            ax = fig.add_subplot(111, projection='3d')
                            scatter = ax.scatter(pca_components_np[:, 0], pca_components_np[:, 1], pca_components_np[:, 2], 
                                                c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
                            ax.set_xlabel('Principal Component 1')
                            ax.set_ylabel('Principal Component 2')
                            ax.set_zlabel('Principal Component 3')
                            fig.colorbar(scatter, ax=ax, label='Cluster')
                            ax.set_title('Clusters in 3D PCA Space')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("PCA components not suitable for 2D/3D visualization (or data empty).")
                except Exception as e:
                    st.error(f"Error generating clustering PCA plot: {e}")
            else:
                st.info("Clustering visualization data not available (e.g., cluster labels or PCA components).")

    else:
        st.info("No visualizations generated yet.") # Or a message that no specific viz data was returned

    st.divider()

    # --- Model Download Section ---
    st.header("Step 4: Model Download") # Adjusted step number for clarity
    if evaluation_report.get('saved_model_path') and os.path.exists(evaluation_report['saved_model_path']):
        try:
            with open(evaluation_report['saved_model_path'], 'rb') as file:
                st.download_button(
                    label="Download Best Model (.pkl)",
                    data=file.read(),
                    file_name=os.path.basename(evaluation_report['saved_model_path']),
                    mime='application/octet-stream'
                )
        except Exception as e:
            st.error(f"Could not prepare model for download: {e}")
    else:
        st.info("No best model found or saved model file not available for download.")

else:
    st.info("Please upload your dataset, select a target column, and confirm the problem type to run the AutoML process.")

# --- About Section ---
st.sidebar.title("About Section")
st.sidebar.info(
    "This AutoML Agent streamlines the machine learning pipeline:\n"
    "- **Data Ingestion:** Upload CSV/JSON files, or connect to databases.\n"
    "- **Target & Problem Type:** Define prediction/analysis goal (Classification, Regression, Clustering).\n"
    "- **Preprocessing:** Handles missing values, scales features, encodes categories.\n"
    "- **Model Training & Tuning:** Compares and optimizes various ML models.\n"
    "- **Evaluation & Reporting:** Provides insights into model performance and important features/clusters.\n\n"
 
)

# --- Footer Section ---

st.markdown(
    """
    <div class="footer">
        <p>Developed by Dhrumil Pawar &copy; 2025. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)