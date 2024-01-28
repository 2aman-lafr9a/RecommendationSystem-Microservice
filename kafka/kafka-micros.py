from kafka import KafkaProducer, KafkaConsumer
import json
import time
import signal
import sys

kafka_bootstrap_servers = 'localhost:9092'
user_interaction_topic = 'user_interaction'
recommendations_topic = 'recommendations_topic'

producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
consumer = KafkaConsumer('recommendations_topic', 
                         bootstrap_servers='localhost:9092', 
                         group_id='consumer_group_id',
                         auto_offset_reset='earliest',
                         enable_auto_commit=False,
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# consumer.subscribe(['recommendations_topic'])

def send_player_data_to_kafka(player_data):
    producer.send(user_interaction_topic, value=player_data)
    print(f"Sent Player Data to Kafka: {player_data}")

def receive_recommendations_from_kafka():
    try:
        for message in consumer:
          print(f"Received message: {message}")
          recommendations = message.value.get('recommendations')
          print(f"Received Recommendations from Kafka: {recommendations}")

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Closing consumer.")
        consumer.close()

# Example player data
player_data_example = {
    "Name": "Anas",
    "Age": 25,
    "Overall": 80,
    "Value(£)": 4000000
}

def signal_handler(sig, frame):
    print("Received Ctrl+C. Exiting gracefully.")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    send_player_data_to_kafka(player_data_example)

    time.sleep(5)

    receive_recommendations_from_kafka()
