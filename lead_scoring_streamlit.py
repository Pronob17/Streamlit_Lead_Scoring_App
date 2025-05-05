import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay

def load_func():
    df_loader = pd.read_csv("Leads.csv")
    return df_loader


def cleaner_func(df_unclean):
    # changing the columns to lower case
    df_unclean.columns = df_unclean.columns.str.replace(" ", "_").str.lower()

    # splitting the target variable and features
    X = df_unclean.drop("converted", axis=1)
    y = df_unclean["converted"]

    # replace the 'Select' with null values
    X.replace("Select", np.nan, inplace=True)

    # feature selection
    X.drop(columns=["prospect_id", "lead_number"], inplace=True, axis=1)

    return X, y


def train_test_split_func(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=17)

    return X_train, X_test, y_train, y_test


def pipeline_func(X_train, X_test, y_train, y_test):
    numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()

    # create numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    # create categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # combine the pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    # model building
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=17))
    ])

    # fit the model
    model_fit = model.fit(X_train, y_train)

    # predict model
    y_prediction = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    return model_fit, y_prediction, y_probability



if __name__ == "__main__":
    # main title
    st.title("LEAD SCORING DASHBOARD")

    # calling the load function
    df = load_func()

    # calling the cleaning function
    X, y = cleaner_func(df)

    # calling train-test-split function
    X_train, X_test, y_train, y_test = train_test_split_func(X, y)

    # pipeline function
    model_fit, y_prediction, y_probability = pipeline_func(X_train, X_test, y_train, y_test)

    # 👇 Sidebar threshold selector
    st.sidebar.subheader("🎯 Choose Lead Scoring Strategy")
    threshold_option = st.sidebar.radio(
        "Scoring Threshold:",
        options=[
            "⚖️ Default (50%)",
            "🚀 Easy Quarter (75%) - High Confidence Leads Only",
            "🔥 Tough Quarter (25%) - Max Outreach"
        ]
    )

    threshold_mapping = {
        "⚖️ Default (50%)": 0.5,
        "🚀 Easy Quarter (75%) - High Confidence Leads Only": 0.75,
        "🔥 Tough Quarter (25%) - Max Outreach": 0.25
    }

    threshold = threshold_mapping[threshold_option]

    # ✅ Apply the selected threshold
    y_prediction = (y_probability >= threshold).astype(int)

    # display lead scores
    lead_scores = (y_probability * 100).round(2)

    # Create a new DataFrame with lead scores, keeping the same index as the original DataFrame
    lead_scores_df = pd.DataFrame({
        "Actual Conversion": y_test,
        "Predicted Lead Scores": lead_scores
    }, index=X_test.index)  # Ensure the index aligns with the original DataFrame

    # Add the predicted lead scores to the original DataFrame
    df_with_scores = df.loc[X_test.index].copy()  # Using X_test's index to align with the original df
    df_with_scores['actual_conversion'] = y_test
    df_with_scores['predicted_lead_scores'] = lead_scores


    # columns creation in streamlit
    st.subheader("🔍 Explore Sample & Original Data")
    st.markdown("Switch between the **two tabs below** to view **random lead scores** and their **original data**.")
    tab1, tab2, tab3 = st.tabs(["10 RANDOM LIST AND THEIR PREDICTIONS", "THE LIST DETAILS", "GRAPHICAL REPRESENTATION"])
    with tab1:
        # Show the updated DataFrame with lead scores

        sample = df_with_scores.sample(10)
        st.dataframe(sample[["lead_number", "actual_conversion", "predicted_lead_scores"]])

    with tab2:
        # Extract lead numbers from the sample
        lead_numbers_list = sample["lead_number"].tolist()

        # Extract the lead numbers and their original indices
        lead_numbers = sample["lead_number"]
        sample_indices = sample.index

        # Use those indices to get rows from original df (this keeps the index and order)

        df_original_matched = df.loc[sample_indices]
        st.dataframe(df_original_matched)
    st.divider()

    with tab3:
        # graphically
        st.header("Graphically:")
        # Showing the sample in the form of a scatter plot
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)

        # Scatter plot with sample data, color-coded by Actual Conversion (0 = No, 1 = Yes)
        scatter = ax.scatter(sample.index, sample["predicted_lead_scores"], c=sample["actual_conversion"], cmap='RdYlGn',
                             s=100,
                             alpha=0.75)

        # Adding a colorbar to represent Actual Conversion values (0 and 1)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Actual Conversion (0 = No, 1 = Yes)')

        # Plot title and labels
        ax.set_title(
            "Lead Score Threshold separates the Predicted Lead Score and 'green/red' color shows the actual conversion")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Predicted Lead Score")

        # Lead score threshold line at y=50
        ax.axhline(y=threshold * 100, color='r', linestyle='--', label='Lead Score Threshold')

        # Adding the legend
        ax.legend()

        # Enabling grid
        ax.grid(True)

        # Adjust layout for better visibility
        fig.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

    st.divider()

    # header for the details
    st.header("STATISTICAL METRICS:")

    # classification report
    st.divider()
    report = classification_report(y_test, y_prediction)
    st.subheader("Classification Report:")
    st.code(report, language='text')

    # roc auc graph
    st.divider()

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model_fit, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    st.pyplot(fig)
    st.markdown(f"#### ***ROC-AUC Score: {roc_auc_score(y_test, y_probability)}***")

    # accuracy score
    st.divider()
    st.subheader("Accuracy Score:")
    st.write(accuracy_score(y_test, y_prediction))

    # confusion matrix
    st.divider()
    st.subheader("Confusion matrix:")
    st.code(confusion_matrix(y_test, y_prediction), language='text')
    st.divider()
