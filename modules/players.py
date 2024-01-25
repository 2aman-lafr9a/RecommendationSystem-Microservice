from cassandra.cluster import Cluster

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

def drop_players_table(session):
    drop_query = "DROP TABLE IF EXISTS players"
    session.execute(drop_query)

    print("Players table dropped successfully")

def create_players_table(session):
    table_query = """
        CREATE TABLE IF NOT EXISTS players (
            Name TEXT,
            Age DOUBLE,
            Overall DOUBLE,
            Club TEXT,
            "Value(£)" DOUBLE,
            Age_Group TEXT,
            Overall_Group TEXT,
            Log_Value DOUBLE,
            Age_Rating DOUBLE,
            Cluster_Labels INT,
            playerId INT PRIMARY KEY
        )
    """
    session.execute(table_query)

    print("Players table created successfully")


def get_max_player_id(session):
    select_query = "SELECT MAX(playerId) as max_id FROM players"
    result = session.execute(select_query)

    max_id = result[0].max_id
    return max_id if max_id else 0

def insert_player_data(session, data):
    
    max_id = get_max_player_id(session)
    data['playerId'] = max_id + 1  
    
    insert_query = f"""
        INSERT INTO players (
            playerId, Name, Age, Overall, Club, "Value(£)", Age_Group,
            Overall_Group, Log_Value, Age_Rating, Cluster_Labels
        )
        VALUES ({data['playerId']}, '{data['Name']}', {data['Age']}, {data['Overall']},
                '{data['Club']}', {data['Value(£)']}, '{data['Age_Group']}', '{data['Overall_Group']}',
                {data['Log_Value']}, {data['Age_Rating']}, {data['Cluster_Labels']})
    """
    session.execute(insert_query)

    print(f"Player data with ID {data['playerId']} inserted successfully")



def query_all_players(session):
    select_query = "SELECT * FROM players"
    result = session.execute(select_query)

    print("All Players:")
    for row in result:
        print(row)

def close_connection(cluster):
    cluster.shutdown()
    print("Connection closed")

if __name__ == "__main__":
    session = connect_to_cassandra()
    # drop_players_table(session)
    create_players_table(session)

    # sample_data = {
    #   # "playerId": 1, 
    #   "Name": "Player 2",
    #   "Age": 25.0,
    #   "Overall": 80.0,
    #   "Club": "Sample Club",
    #   "Value(£)": 4000000.0,
    #   "Age_Group": "Sample Age Group",
    #   "Overall_Group": "Sample Overall Group",
    #   "Log_Value": 123.45,
    #   "Age_Rating": 4.5,
    #   "Cluster_Labels": 1, 
    # }
    # insert_player_data(session, sample_data)

    query_all_players(session)

    close_connection(session.cluster)
