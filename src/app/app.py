from flask import Flask, request, jsonify
from ..model.recommendation_system import recommendation_system

app = Flask(__name__)

# Include the recommendation_system function here

@app.route('/recommend', methods=['POST'])
def recommend():
    player_data = request.json.get('player_data')

    if not player_data:
        return jsonify({'error': 'Player data not provided in the request'}), 400

    recommendations, csv_filename = recommendation_system(player_data)

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
