from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# pip install apache-airflow-providers-apache-livy

default_args = {
  'owner': 'coder2j',
  'retries': 5,
  'retry_delay': timedelta(minutes=2)
}

with DAG(
  dag_id='our_first_dag_v5',
  default_args=default_args,
  description="This is our first dag we create",
  start_date=datetime(2024, 1, 17, 5),
  schedule_interval='@daily'
) as dag:
  task1 = BashOperator(
    task_id='first_task',
    bash_command="echo hello Said, Anas, This the first task 1!"
  )

  task2 = BashOperator(
    task_id='second_task',
    bash_command="echo hello world, Anas, This the second task 2!"
  )

  task3 = BashOperator(
    task_id='third_task',
    bash_command="echo hello world, Anas, This the third task 4!"
  )

  task1 >> task2 >> task3