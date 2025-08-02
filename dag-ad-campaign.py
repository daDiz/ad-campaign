import json
from datetime import datetime

import gcsfs
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from google.cloud import bigquery, logging, storage
from imblearn.over_sampling import RandomOverSampler

from ad_campaign_train import (
    evaluate_model,
    load_rev_data,
    load_spend_data,
    merge_data,
    preprocess_rev_data,
    preprocess_spend_data,
    save_model,
    train_model,
    write_metrics_to_bigquery,
)

logging_client = logging.Client()
logger = logging_client.logger("ad-campaign-training-logs")


def validate_csv():
    # Load data
    fs = gcsfs.GCSFileSystem()
    with fs.open(
        "gs://my-first-project-466020-bucket/advertising_roi_data/campaign_spend.csv"
    ) as f:
        df_spend = pd.read_csv(f)

    # Define expected columns
    expected_spend_cols = [
        "CAMPAIGN",
        "CHANNEL",
        "DATE",
        "TOTAL_CLICKS",
        "TOTAL_COST",
        "ADS_SERVED",
    ]

    # Check if the loaded columns are same as expected columns
    if list(df_spend.columns) == expected_spend_cols:
        return True
    else:
        logger.log_struct(
            {
                "keyword": "Ad_Campaign_Model_Training",
                "description": "This log captures the last run for Model Training",
                "training_timestamp": datetime.now().isoformat(),
                "model_output_msg": "Input Data is not valid",
                "training_status": 0,
            }
        )
        raise ValueError(
            f"CSV does not have expected columns. Columns in CSV are: {list(df_spend.columns)}"
        )


def read_last_training_metrics():
    client = bigquery.Client()
    table_id = "my-first-project-466020.ml_ops.advertising_roi_model_metrics"
    query = f"""
        SELECT *
        FROM `{table_id}`
        where algo_name='linear_regression'
        ORDER BY training_time DESC
        LIMIT 1
    """
    result = client.query(query).result()
    latest_row = next(result)
    return json.loads(latest_row[2])


def train_evaluate_model():
    # Load data for evaluation
    df_spend = load_spend_data()
    df_spend = preprocess_spend_data(df_spend)
    df_rev = load_rev_data()
    df_rev = preprocess_rev_data(df_rev)
    df = merge_data(df_rev, df_spend)

    # Train the model on new data
    model, X_train, X_test, y_train, y_test = train_model(df)

    # Get the current model metrics for evaluation
    r2_score_train, r2_score_test = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    model_metrics = {"r2_train": r2_score_train, "r2_test": r2_score_test}
    model_name = "linear_regression"
    training_time = datetime.now()

    # Get the last/existing model metrics for comparison
    last_model_metrics = read_last_training_metrics()
    last_r2_train = last_model_metrics["r2_train"]
    last_r2_test = last_model_metrics["r2_test"]

    # Define the threshold values for precision and recall
    r2_train_threshold = 0.9
    r2_test_threshold = 0.8

    # Save the model artifact if metrics are above the thresholds
    if (
        r2_score_train >= r2_train_threshold
        and r2_score_test >= r2_test_threshold
        and (r2_score_train >= last_r2_train and r2_score_test >= last_r2_test)
    ):
        save_model(model)
        write_metrics_to_bigquery(model_name, training_time, model_metrics)
        logger.log_struct(
            {
                "keyword": "Ad_Campaign_Model_Training",
                "description": "This log captures the last run for Model Training",
                "training_timestamp": datetime.now().isoformat(),
                "model_output_msg": "Model artifact saved",
                "training_status": 1,
            }
        )
    else:
        logger.log_struct(
            {
                "keyword": "Ad_Campaign_Model_Training",
                "description": "This log captures the last run for Model Training",
                "training_timestamp": datetime.now().isoformat(),
                "model_output_msg": "Model metrics do not meet the defined threshold",
                "model_metrics": model_metrics,
                "training_status": 0,
            }
        )


# Define the default_args dictionary
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 1,
}

# Instantiate the DAG
dag = DAG(
    "dag_ad_campaign_continuous_training",
    default_args=default_args,
    description="Ad campaign training DAG",
    schedule_interval=None,
)

# Define the tasks/operators
validate_csv_task = PythonOperator(
    task_id="validate_csv",
    python_callable=validate_csv,
    dag=dag,
)

train_evaluate_task = PythonOperator(
    task_id="model_training_evaluation",
    python_callable=train_evaluate_model,
    dag=dag,
)

validate_csv_task >> train_evaluate_task
