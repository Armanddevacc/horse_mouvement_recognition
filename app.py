from flask import Flask, request, jsonify, abort
from functools import wraps
from ModelePrediction import ModelePrediction



app = Flask(__name__)

# Décorateur pour vérifier l'authentification
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')  # Récupère le token depuis les en-têtes
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        if token != 'your_secure_token':  # Remplace 'your_secure_token' par le token de ton choix
            return jsonify({'message': 'Invalid token!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@token_required
def predict():
    # Charger ton modèle
    mon_modele = ModelePrediction('data/model_cheveux.keras')

    data = request.get_json(force=True)  # JSON reçu par l'application
    # Fonction que je dois définir pour traiter les données et les adapter
    predictions = mon_modele.predire(data)
    return jsonify({'predictions': predictions.tolist()})






# @app.route("/login", methods=["POST"])
# def login():
#     """Login a spectator and provide a token for future requests to the API

#     Returns
#     -------
#     data
#         a token to authenticate future request to the API.
#         an error message "No username or password provided" if the
#             username or password is not provided
#         an error message "Error: while authenticating spectator" if an error
#             occured while authenticating the spectator.
#     status_code
#         200 if the token is correctly provided
#         400 if the username or password is not provided
#         500 if an error occured while authenticating the spectator
#     """
    
#     # Récupére données
#     data = request.get_json()

#     # Vérifier si le nom d'utilisateur et le mot de passe sont fournis
#     if 'username' not in data or 'password' not in data:
#         return jsonify({"error": "No username or password provided"}), 400

#     username = data['username']
#     password = data['password']
#     try:
#         # Vérifier le mot de passe (comparaison entre le mot de passe haché et celui fourni)
#         if not check_spectator(username,password):
#             return jsonify({"error": "Invalid username or password"}), 401


#         # Générer le token avec une expiration de 1 heure
#         token = generate_token(username)

#         # Retourner le token dans la réponse
#         return jsonify({"token": token}), 200

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": "Error: while authenticating spectator"}), 500



if __name__ == '__main__':
    mon_modele = ModelePrediction('data/model_cheveux.keras')

    predictions = mon_modele.afficher_matrice_confusion()
    app.run(debug=True)