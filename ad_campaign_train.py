import json
from datetime import datetime

import joblib
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

storage_client = storage.Client()
bucket = storage_client.bucket("my-first-project-466020-bucket")


def load_spend_data():
    # df = pd.read_csv("data/campaign_spend.csv")
    df = pd.read_csv(
        "gs://my-first-project-466020-bucket/advertising_roi_data/campaign_spend.csv"
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    return df


def load_rev_data():
    # df = pd.read_csv("data/monthly_revenue.csv")
    df = pd.read_csv(
        "gs://my-first-project-466020-bucket/advertising_roi_data/monthly_revenue.csv"
    )
    return df


def preprocess_spend_data(df):
    grouped = df.groupby(by=["YEAR", "MONTH", "CHANNEL"], as_index=False)[
        "TOTAL_COST"
    ].sum()
    grouped = grouped.sort_values(by=["YEAR", "MONTH"])
    df_spend_per_month = pd.pivot_table(
        grouped, index=["YEAR", "MONTH"], columns="CHANNEL", aggfunc="sum"
    )
    df_spend_per_month = df_spend_per_month.rename(
        columns={
            "email": "EMAIL",
            "search_engine": "SEARCH_ENGINE",
            "social_media": "SOCIAL_MEDIA",
            "video": "VIDEO",
        }
    )
    df_spend_per_month = df_spend_per_month.reset_index()
    df_spend_per_month.columns = [
        "_".join(filter(None, col)) for col in df_spend_per_month.columns.values
    ]
    df_spend_per_month = df_spend_per_month.rename(
        columns={
            "TOTAL_COST_EMAIL": "EMAIL",
            "TOTAL_COST_SEARCH_ENGINE": "SEARCH_ENGINE",
            "TOTAL_COST_SOCIAL_MEDIA": "SOCIAL_MEDIA",
            "TOTAL_COST_VIDEO": "VIDEO",
        }
    )
    return df_spend_per_month


def preprocess_rev_data(df):
    df_rev_per_month = df.groupby(by=["YEAR", "MONTH"], as_index=False)["REVENUE"].sum()
    df_rev_per_month = df_rev_per_month.sort_values(by=["YEAR", "MONTH"])
    return df_rev_per_month


def merge_data(df_rev, df_spend):
    df_merged = df_rev.merge(df_spend, on=["YEAR", "MONTH"])
    df_merged = df_merged.dropna()
    df_merged = df_merged.drop(["YEAR", "MONTH"], axis=1)
    return df_merged


def train_model(df):
    numeric_feats = [
        "EMAIL",
        "SEARCH_ENGINE",
        "SOCIAL_MEDIA",
        "VIDEO",
    ]
    polynomial_feat_degree = 2

    numeric_transformer = Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(degree=polynomial_feat_degree, include_bias=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_feats)])

    model = Pipeline(
        [("preprocessor", preprocessor), ("regression", LinearRegression())]
    )

    X, y = df.drop("REVENUE", axis=1), df["REVENUE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    r2_score_train = model.score(X_train, y_train)
    r2_score_test = model.score(X_test, y_test)
    return r2_score_train, r2_score_test


def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    client = bigquery.Client()
    table_id = "my-first-project-466020.ml_ops.advertising_roi_model_metrics"
    table = bigquery.Table(table_id)

    row = {
        "algo_name": algo_name,
        "training_time": training_time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_metrics": json.dumps(model_metrics),
    }

    errors = client.insert_rows_json(table, [row])
    if errors == []:
        print("Metrics inserted into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery: ", errors)


def save_model(model):
    artifact_name = "model.joblib"
    joblib.dump(model, artifact_name)
    model_artifact = bucket.blob("advertising_roi_artifact/" + artifact_name)
    model_artifact.upload_from_filename(artifact_name)


def main():
    df_spend = load_spend_data()
    df_spend = preprocess_spend_data(df_spend)
    df_rev = load_rev_data()
    df_rev = preprocess_rev_data(df_rev)
    df = merge_data(df_rev, df_spend)

    model, X_train, X_test, y_train, y_test = train_model(df)

    r2_score_train, r2_score_test = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    model_metrics = {"r2_train": r2_score_train, "r2_test": r2_score_test}
    model_name = "linear_regression"
    training_time = datetime.now()

    print(model_metrics)
    write_metrics_to_bigquery(model_name, training_time, model_metrics)

    save_model(model)


if __name__ == "__main__":
    main()
