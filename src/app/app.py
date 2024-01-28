from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
socketio = SocketIO(app)

kafka_config = {
    'bootstrap_servers': 'localhost:9092',
    'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
}

producer = KafkaProducer(**kafka_config)

def send_to_kafka_recommendations(recommendations):
    topic = 'recommendations_topic'
    recommendations = json.loads(json.dumps(recommendations, default=str))
    message = {
        'recommendations': recommendations,
    }
    producer.send(topic, value=message)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@app.route('/recommend', methods=['POST'])
def recommend():
    player_data = request.json.get('player_data')

    print("-------Player Data-------")
    print(player_data)
    print("--------------------------------")

    if not player_data:
        return jsonify({'error': 'Player data not provided in the request'}), 400

    recommendations, csv_filename = recommendation_system(player_data)
    send_to_kafka_recommendations(recommendations)

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
