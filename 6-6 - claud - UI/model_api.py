from flask import Flask, request, jsonify
import torch
from model import LoanClassifier  # Make sure model.py is in the same folder

app = Flask(__name__)

# Load your trained model here (adjust as needed)
model = LoanClassifier()
model.load_state_dict(torch.load('./saved_models/global_model_final.pth', map_location='cpu'))
model.eval()
print("Loaded model weights from loan_classifier.pth")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')
    if not features or len(features) != 11:
        return jsonify({'error': 'features must be an array of 11 numbers'}), 400
    with torch.no_grad():
        input_tensor = torch.tensor([features], dtype=torch.float32)
        output = model(input_tensor)
        probability = float(output.item())
        prediction = "Approved" if probability > 0.5 else "Rejected"
        print("Input features:", features)
        print("Model raw output:", output)
        print("Probability:", probability)
        print("Prediction:", prediction)
    return jsonify({
        'prediction': prediction,
        'probability': f"{probability:.4f}",
        'details': f"Model prediction for features: {features}"
    })

if __name__ == '__main__':
    app.run(port=5000)