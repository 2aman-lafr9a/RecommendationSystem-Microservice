from flask import Flask, request, jsonify
from kafka import KafkaProducer
from ..model.recommendation_system import recommendation_system
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

app = Flask(__name__)


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

    message = {
        'recommendations': recommendations,
    }

    producer.send(topic, value=message)


send_to_kafka_recommendations("Hello Anas !!")

@app.route('/recommend', methods=['POST'])
def recommend():
    player_data = request.json.get('player_data')

    if not player_data:
        return jsonify({'error': 'Player data not provided in the request'}), 400

    # Call the recommendation_system function with player_data
    recommendations, csv_filename = recommendation_system(player_data)

    # Send recommendations to Kafka
    # send_to_kafka_recommendations(recommendations)

    # Return the recommendations as JSON
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
