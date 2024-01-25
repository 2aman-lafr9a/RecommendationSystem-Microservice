from cassandra.cluster import Cluster
from datetime import datetime

def connect_to_cassandra():
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()

    # Create a keyspace
    keyspace_query = "CREATE KEYSPACE IF NOT EXISTS my_keyspace WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}"
    session.execute(keyspace_query)

    # Use the keyspace
    session.set_keyspace('my_keyspace')

    print("Connected to Cassandra")
    return session

def drop_insurances_table(session):
    drop_query = "DROP TABLE IF EXISTS insurances"
    session.execute(drop_query)

    print("Insurances table dropped successfully")

def create_insurances_table(session):
    table_query = """
        CREATE TABLE IF NOT EXISTS insurances (
            insuranceId INT PRIMARY KEY,
            type TEXT,
            description TEXT,
            price INT,
            name TEXT,
            created_at TIMESTAMP
        )
    """
    session.execute(table_query)

    print("Insurances table created successfully")

def get_max_insurance_id(session):
    select_query = "SELECT MAX(insuranceId) as max_id FROM insurances"
    result = session.execute(select_query)

    max_id = result[0].max_id
    return max_id if max_id else 0

def insert_insurance_data(session, data):
    max_id = get_max_insurance_id(session)
    data['insuranceId'] = max_id + 1

    insert_query = f"""
        INSERT INTO insurances (
            insuranceId, type, description, price, name, created_at
        )
        VALUES ({data['insuranceId']}, '{data['type']}', '{data['description']}', {data['price']},
                '{data['name']}', '{data['created_at']}')
    """
    session.execute(insert_query)

    print(f"Insurance data with ID {data['insuranceId']} inserted successfully")

def query_all_insurances(session):
    select_query = "SELECT * FROM insurances"
    result = session.execute(select_query)

    print("All Insurances:")
    for row in result:
        print(row)

def close_connection(cluster):
    cluster.shutdown()
    print("Connection closed")

if __name__ == "__main__":
    session = connect_to_cassandra()
    # drop_insurances_table(session)
    create_insurances_table(session)

    sample_data = {
        # "insuranceId": 1,
        "type": "Normal",
        "description": "Sample Description",
        "price": 2000,
        "name": "Insurance A",
        "created_at": datetime.now()
    }
    insert_insurance_data(session, sample_data)

    query_all_insurances(session)

    close_connection(session.cluster)
