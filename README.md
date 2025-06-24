# 🤖 Intelligent AutoML Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL_HERE) [![Render App](https://img.shields.io/badge/Deploy%20on-Render-46E3B7?style=flat&logo=render)](YOUR_RENDER_DEMO_URL_HERE) [![GitHub last commit](https://img.shields.io/github/last-commit/dhrumil1128/AutoML_Agent)](https://github.com/dhrumil1128/AutoML_Agent/commits/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Detailed Overview

The **Intelligent AutoML Agent** is a cutting-edge, interactive web application meticulously crafted to democratize the power of machine learning. Leveraging the simplicity of Streamlit for its intuitive front-end and the robust capabilities of Scikit-learn for its analytical backbone, this agent provides an automated, end-to-end solution for common machine learning tasks.

From raw data ingestion (supporting both file uploads and direct database connections) to sophisticated preprocessing, comprehensive model training, exhaustive hyperparameter tuning, insightful evaluation, and even model export, this agent streamlines the entire ML pipeline. It intelligently handles various complexities such as missing values, categorical encoding, and feature scaling, allowing users to focus on problem definition rather than intricate technical details.

Designed with both efficiency and user experience in mind, the Intelligent AutoML Agent empowers a wide range of users – from aspiring data scientists and domain experts without extensive coding experience to seasoned professionals seeking rapid prototyping – to build, analyze, and deploy high-performing machine learning models with unprecedented ease and speed. It serves as a personal AI-powered data scientist, abstracting away the underlying complexities to deliver actionable insights and predictive capabilities.

## The Problem It Solves

Building a robust machine learning model involves several complex and often time-consuming steps:

1.  **Data Preparation is Tedious:** Cleaning, transforming, and preparing raw data (handling missing values, encoding categorical features, scaling numerical data) is time-consuming and prone to errors.
2.  **Overwhelming Model Choices:** Deciding which algorithm (e.g., Logistic Regression vs. Random Forest vs. SVM) is best suited for a specific problem (classification, regression) can be daunting.
3.  **Complex Hyperparameter Tuning:** Optimizing a chosen model's performance requires extensive knowledge of its internal parameters and iterative experimentation, often consuming substantial computational resources and time.
4.  **Evaluation and Interpretation Gaps:** Understanding and communicating model performance beyond a single metric, along with interpreting *why* a model makes certain predictions (e.g., feature importance), can be difficult.
5.  **Accessibility Barrier:** The steep learning curve of programming languages and ML frameworks limits access to powerful predictive analytics for many potential users.

These barriers prevent effective and widespread adoption of machine learning solutions.

## Our Solution: The Intelligent AutoML Agent

This agent directly addresses these pain points by offering an automated, integrated, and user-friendly platform:

* **Simplified Data Ingestion:** Effortlessly upload CSV files or establish direct connections to various SQL databases (SQLite, PostgreSQL, MySQL, SQL Server) by simply providing connection details and SQL queries.
* **Automated Preprocessing:** Intelligently handles common data preparation steps, including imputation for missing numerical (mean) and categorical (most frequent) values, One-Hot Encoding for categorical features, and StandardScaler for numerical features – all done behind the scenes.
* **Smart Problem Type Detection:** Automatically suggests "Classification" or "Regression" based on the characteristics of the target column, with an option for manual override.
* **Extensive Model Support & Advanced Tuning:**
    * **Classification:** Auto-trains and tunes Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, Support Vector Classifier, Decision Tree Classifier, and K-Nearest Neighbors Classifier.
    * **Regression:** Auto-trains and tunes Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, Decision Tree Regressor, and K-Nearest Neighbors Regressor.
    * Utilizes `GridSearchCV` for automated and exhaustive hyperparameter optimization across all candidate models to identify the best configuration.
* **Comprehensive Evaluation & Reporting:** Provides detailed performance metrics (Accuracy, F1-Score, Precision, Recall for Classification; MSE, R2 for Regression) for the best model on unseen test data, along with a comparative summary of all trained models.
* **Insightful Visualizations:** Generates crucial plots such as Feature Importance Bar Charts (for tree-based models), Confusion Matrices (for Classification problems), and Predicted vs. Actual Plots (for Regression problems) to aid in model understanding and explainability.
* **Direct Model Export:** Enables users to download the best-trained machine learning model (encapsulating its entire preprocessing pipeline) as a `.pkl` file, making it immediately ready for integration into other applications or deployment scenarios.
* **Intuitive & Responsive UI:** Built entirely with Streamlit, providing a clean, interactive, and highly responsive web interface that enhances the user experience and makes complex ML workflows accessible.
* **Theme Adaptability:** Ensures a seamless and visually pleasing user experience in both Streamlit's light and dark themes.

## Key Features

* **Data Ingestion:** CSV file upload, SQL database connection (SQLite, PostgreSQL, MySQL, SQL Server).
* **Automated Data Preprocessing:** Missing value imputation (mean/mode), One-Hot Encoding, StandardScaler.
* **Problem Type Detection:** Automatic suggestion (Classification/Regression) with manual override.
* **Diverse Model Training:** Supports a wide range of popular classification and regression algorithms.
* **Hyperparameter Tuning:** Automated optimization using `GridSearchCV`.
* **Comprehensive Evaluation:** Key metrics for both classification and regression.
* **Visualizations:** Feature Importance, Confusion Matrix, Predicted vs. Actual plots.
* **Model Export:** Download best model as `.pkl` for easy deployment.
* **Interactive UI:** Powered by Streamlit for ease of use.
* **Theme Support:** Seamless experience in light and dark modes.

## How It Works (Under the Hood)

The core automation logic is encapsulated within `automl_core.py`, orchestrating a robust machine learning pipeline:

1.  **Data Splitting:** The input dataset is intelligently split into training and testing sets to ensure unbiased model evaluation.
2.  **Automated Preprocessing Pipeline:**
    * It identifies numerical and categorical features.
    * Separate preprocessing steps (imputation, scaling for numerical; imputation, one-hot encoding for categorical) are defined.
    * These are combined into a powerful `ColumnTransformer` within a `scikit-learn Pipeline`.
3.  **Iterative Model Training & Selection:**
    * A curated list of machine learning models (e.g., Logistic Regression, Random Forest) is prepared for the detected `problem_type`.
    * For each model, a comprehensive `Pipeline` is constructed, integrating the preprocessor and the model itself.
    * `GridSearchCV` is then employed to systematically search for the optimal hyperparameters for each model using cross-validation.
    * The model (with its best hyperparameters) that achieves the highest performance metric (e.g., Accuracy for classification, R2 for regression) on cross-validation is selected as the `best_model`.
4.  **Rigorous Evaluation & Visualization:** The chosen `best_model` is rigorously evaluated on the unseen test data. Detailed performance metrics are computed, and data for insightful visualizations (like feature importances and confusion matrices) is generated.
5.  **Persistent Model Saving:** The final `best_model` object, which includes all the learned preprocessing transformations, is serialized using Python's `pickle` library. This `.pkl` file allows the model to be easily loaded and used for predictions in new environments without retraining.

## Getting Started (Local Setup)

Follow these steps to get your Intelligent AutoML Agent up and running on your local machine.

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dhrumil1128/AutoML_Agent.git](https://github.com/dhrumil1128/AutoML_Agent.git)
    cd AutoML_Agent
    ```

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure you have a `requirements.txt` file in the root of your repository listing all necessary libraries. If you don't have one, generate it first:
    ```bash
    pip freeze > requirements.txt
    ```
    Then install all required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Common dependencies include `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `sqlalchemy`. If you plan to use specific database types, you might need their respective drivers, e.g., `psycopg2-binary` for PostgreSQL, `pymysql` for MySQL, `pyodbc` for SQL Server.)*

### Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run agent.py
    ```
2.  Your default web browser should automatically open the application at `http://localhost:8501`.

### Testing SQL Database Import (SQLite Example)

To test the robust database connection feature using a local SQLite file:

1.  **Install SQLite Extension in VS Code:** Open VS Code, go to the Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X`), and search for and install the "SQLite" extension by `alexcvzz`.
2.  **Create a New Database File:** In VS Code, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type `SQLite: Open Database`, and enter a new file name (e.g., `my_test_data.db`) within your project folder. Save it.
3.  **Add Sample Test Data:** In the SQLite Explorer sidebar, right-click on your newly created `my_test_data.db` file and select "New Query." Paste the following SQL statements into the new query tab:
    ```sql
    CREATE TABLE IF NOT EXISTS sales_data (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_category TEXT NOT NULL,
        region TEXT NOT NULL,
        price REAL NOT NULL,
        quantity INTEGER NOT NULL,
        discount REAL DEFAULT 0.0,
        is_promoted INTEGER DEFAULT 0, -- Example target for classification
        revenue REAL -- Example target for regression
    );

    INSERT INTO sales_data (product_category, region, price, quantity, discount, is_promoted, revenue) VALUES
    ('Electronics', 'North', 1200.00, 1, 0.1, 1, 1080.00),
    ('Apparel', 'South', 50.00, 5, 0.0, 0, 250.00),
    ('Electronics', 'East', 800.00, 2, 0.05, 1, 1520.00),
    ('Home Goods', 'West', 150.00, 3, 0.0, 0, 450.00),
    ('Apparel', 'North', 75.00, 4, 0.15, 0, 255.00);

    -- Verify the data has been added
    SELECT * FROM sales_data;
    ```
    Select all the SQL statements and execute them by right-clicking and choosing "Run Query" (or `Ctrl+Enter`).
4.  **Connect in Streamlit App:**
    * Go to your running Streamlit app (`http://localhost:8501`).
    * In "Step 1: Ingest Your Dataset," select the "Connect to Database (SQL Query)" option.
    * Choose `SQLite` as the "Database Type".
    * For the "Database Name" field, enter the **full absolute path** to your `my_test_data.db` file (e.g., `C:\Users\YourUser\YourProject\my_test_data.db` on Windows, or `/home/youruser/yourproject/my_test_data.db` on Linux/macOS).
    * In the "SQL Query" text area, type `SELECT * FROM sales_data;`.
    * Click the "Fetch Data from Database" button.
    * The app should display "Data fetched successfully from database!" and a preview of the `sales_data` table.

## Deployment

This application is designed for straightforward deployment, allowing you to share your powerful AutoML Agent with others:

* **Render:** A popular cloud platform that simplifies the deployment of web services, including Streamlit applications. You can host your agent directly from your GitHub repository.
* **Streamlit Community Cloud:** The easiest and fastest way to deploy Streamlit applications directly from your GitHub repository for free. This is ideal for showcasing your app quickly.

## Future Enhancements

While feature-complete, this AutoML Agent can be continuously improved. Here are some ideas for future enhancements:

* **Advanced Models:** Integrate support for deep learning frameworks (e.g., TensorFlow, PyTorch) for more complex tasks.
* **Feature Engineering:** Add automated feature generation or selection techniques.
* **Explainable AI (XAI):** Incorporate tools like SHAP or LIME for deeper model interpretability beyond just feature importance.
* **Time Series & NLP Support:** Extend the agent's capabilities to handle time-series forecasting or natural language processing tasks.
* **MLOps Integration:** Incorporate experiment tracking (MLflow), model versioning, and automated retraining pipelines.
* **Real-time Prediction API:** Develop a separate Flask/FastAPI service that consumes the downloaded `.pkl` model for real-time predictions via an API endpoint.
* **User Authentication:** Implement basic user login for personalized experiences or data privacy.

---

## Demo Link

Experience the Intelligent AutoML Agent live in action:

[**Launch the Streamlit App Demo**](YOUR_STREAMLIT_APP_URL_HERE) --- https://automl-agent.onrender.com

## Contact

I'm **Dhrumil Pawar**, the developer behind this project. I'm passionate about building intelligent systems and solving real-world problems with data science and machine learning. Feel free to connect or reach out!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dhrumil-pawar/)

