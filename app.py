

from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('fashion_mnist_cnn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    
    image_data = request.json['image_data']
    
    
    processed_image = preprocess_image(image_data)

    # Making prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # Returning prediction
    return jsonify({'predicted_class': int(predicted_class)})

def preprocess_image(image_data):
   
    processed_image = image_data 
    return processed_image

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
