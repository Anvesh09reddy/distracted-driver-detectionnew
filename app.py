from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'C:\Users\Anvesh\Downloads\mini project\model\model.h5')

# Define class labels mapping
class_labels = {
    0: "No distraction",
    1: "Texting",
    2: "Talking on the phone",
    3: "Reaching for something",
    4: "Eating",
    5: "Drinking",
    6: "Grooming",
    7: "Looking at something else",
    8: "Looking at the passenger",
    9: "Sleeping"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the 'file' key is in the request
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    # Save the uploaded file to a specific directory
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Process the image for prediction
    image = Image.open(file_path)
    image = image.resize((224, 224))  # Resize the image to the input shape of your model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    return f'The predicted class is: {predicted_class}, which corresponds to "{predicted_label}".'

if __name__ == '__main__':
    app.run(debug=True)

