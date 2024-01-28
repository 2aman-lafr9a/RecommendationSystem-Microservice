from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from recommendation_sys.recommendation_system import recommendation_system

app = Flask(__name__)

kafka_config = {
    'bootstrap_servers': 'localhost:9092',  
    'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
}

# Kafka producer and consumer instance
producer = KafkaProducer(**kafka_config)

player_data_consumer = KafkaConsumer('user_interaction', group_id='flask_microservice', auto_offset_reset='earliest',
                                     enable_auto_commit=False, value_deserializer=lambda x: json.loads(x.decode('utf-8')))


def send_to_kafka_recommendations(recommendations):
    """
    Send recommendations to Kafka topic.
    """
    topic = 'recommendations_topic'  

    recommendations = json.loads(json.dumps(recommendations, default=str))

    message = {
        'recommendations': recommendations,
    }

    producer.send(topic, value=message)


# send_to_kafka_recommendations("Hello Anas !!")


def process_player_data_and_send_recommendations(player_data):
    """
    Process player data and send recommendations to Kafka topic.
    """
    recommendations, csv_filename = recommendation_system(player_data)
    send_to_kafka_recommendations(recommendations)



def receive_from_kafka_player_data():
    for message in player_data_consumer:
        # player_data = message.value.get('player_data')
        player_data = message.value
        print("Received Player Data from Kafka:", player_data)

        process_player_data_and_send_recommendations(player_data)


if __name__ == '__main__':
    receive_from_kafka_player_data()

    app.run(host='0.0.0.0', port=5000)




# @app.route('/recommend', methods=['POST'])
# def recommend():
#     player_data = request.json.get('player_data')

#     print("-------Palayer Data-------")
#     print(player_data)
#     print("--------------------------------")

#     if not player_data:
#         return jsonify({'error': 'Player data not provided in the request'}), 400

#     # Call the recommendation_system function with player_data
#     recommendations, csv_filename = recommendation_system(player_data)

#     send_to_kafka_recommendations(recommendations)

#     return jsonify({'recommendations': recommendations})
#     # return jsonify({'msg': "Hello world!"})