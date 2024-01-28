
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.data_preprocessing.preprocessing_script import data_preprocessing
from src.training.model_training import train_model

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='recommendation_system_dag',
    start_date=datetime(2024, 1, 17, 5),
    default_args=default_args,
    description='DAG for Recommendation System.',
    schedule_interval='@daily'
) as dag:

    # Task 1: Preprocessing Task
    preprocessing_task = PythonOperator(
        task_id='preprocessing_task',
        python_callable=data_preprocessing,  
        provide_context=True,
        # dag=dag,
    )

    # Task 2: Training Model Task
    training_task = PythonOperator(
        task_id='training_task',
        python_callable=train_model,  
        provide_context=True,
        # dag=dag,
    )

    # Task 3: Move Images to Appropriate Directory
    move_images_task = BashOperator(
        task_id='move_images_task',
        bash_command=f'mv /../../images/*.png /../../images_backup/',
        # dag=dag,
    )


    preprocessing_task >> training_task >> move_images_task

# if __name__ == "__main__":
#     dag.cli()
    

    
# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.bash import BashOperator

# pip install apache-airflow-providers-apache-livy



# default_args = {
#   'owner': 'coder2j',
#   'retries': 5,
#   'retry_delay': timedelta(minutes=2)
# }

# with DAG(
#   dag_id='our_first_dag_v6',
#   default_args=default_args,
#   description="This is our first dag we create",
#   start_date=datetime(2024, 1, 17, 5),
#   schedule_interval='@daily'
# ) as dag:
#   task1 = BashOperator(
#     task_id='first_task',
#     bash_command="echo hello Said, Anas, This the first task 1!"
#   )

#   task2 = BashOperator(
#     task_id='second_task',
#     bash_command="echo hello world, Anas, This the second task 2!"
#   )

#   task3 = BashOperator(
#     task_id='third_task',
#     bash_command="echo hello world, Anas, This the third task 4!"
#   )

#   task1 >> task2 >> task3