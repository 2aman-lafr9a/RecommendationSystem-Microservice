from kafka import KafkaConsumer
import json

kafka_config = {
    'bootstrap_servers': 'localhost:9092', 
    'group_id': 'test_consumer_group', 
    'auto_offset_reset': 'earliest',
    'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
}

consumer = KafkaConsumer('recommendations_topic', **kafka_config)

for message in consumer:
    print(f"Received message: {message.value}")

consumer.close()
