from flask import Flask, request, jsonify
from flask_cors import CORS
from kafka import KafkaProducer
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from recommendation_sys.recommendation_system import recommendation_system

app = Flask(__name__)

CORS(app)

kafka_config = {
    'bootstrap_servers': 'localhost:9092',  
    'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
}

# Kafka producer instance
producer = KafkaProducer(**kafka_config)

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

@app.route('/recommend', methods=['POST'])
def recommend():
    player_data = request.json.get('player_data')

    print("-------Palayer Data-------")
    print(player_data)
    print("--------------------------------")

    if not player_data:
        return jsonify({'error': 'Player data not provided in the request'}), 400

    # Call the recommendation_system function with player_data
    recommendations, csv_filename = recommendation_system(player_data)

    send_to_kafka_recommendations(recommendations)

    return jsonify({'recommendations': recommendations})
    # return jsonify({'msg': "Hello world!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
