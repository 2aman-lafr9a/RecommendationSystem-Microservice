import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def connect_to_database(host, port, user, password, database):
    db_params = { 
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'database': database
    }

    try:
        conn = psycopg2.connect(**db_params)

        test_query = "SELECT 1"
        with conn.cursor() as cursor:
            cursor.execute(test_query)
            result = cursor.fetchone()

        if result:
            print(f"Connected to the database '{database}' on host '{host}' successfully.")
            return conn
        else:
            print(f"Failed to execute the test query. Please check your database settings.")
            return None

    except Exception as e:
        print(f"Failed to connect to the database '{database}' on host '{host}': {str(e)}")
        return None

def extract_ratings_data(conn):
    ratings_query = "SELECT * FROM ratings"

    # Use pandas to read the result of the SQL query into a DataFrame
    df_ratings = pd.read_sql_query(ratings_query, conn)
    return df_ratings

def extract_insurances_data(conn):
    insurances_query = "SELECT * FROM offers"

    # Use pandas to read the result of the SQL query into a DataFrame
    df_insurances = pd.read_sql_query(insurances_query, conn)
    return df_insurances

def close_connection(conn):
    conn.close()
