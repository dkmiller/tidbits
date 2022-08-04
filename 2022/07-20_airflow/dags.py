from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


with DAG(
    "dan_test_2",
    description="Learn Airflow (take 2)",
    # https://stackoverflow.com/a/57954990
    is_paused_upon_creation=False,
    schedule_interval=timedelta(minutes=1),
    start_date=datetime(2022, 7, 19),
    tags=["tidbits"],
) as dag:
    bo2 = BashOperator(task_id="print_date", bash_command="date")
