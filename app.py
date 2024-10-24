from flask import Flask, request, jsonify
import ModelePrediction


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Charger ton modèle
    mon_modele = ModelePrediction('data/model_cheveux.keras')

    data = request.get_json(force=True)  # JSON reçu par l'application
    # Fonction que je dois définir pour traiter les données et les adapter
    predictions = mon_modele.predire(data)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
