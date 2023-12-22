from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    return jsonify({'recommendations': [...]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)