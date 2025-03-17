import os
from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import io

# Initialize Flask app and set template & static folders
app = Flask(__name__, template_folder="templates", static_folder="static")

# Function to process the image
def process_image(image):
    # Resize the image
    image_resized = cv2.resize(image, None, fx=0.5, fy=0.5)

    # Remove impurities from the image
    image_cleared = cv2.medianBlur(image_resized, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)
    image_cleared = cv2.edgePreservingFilter(image_cleared, sigma_s=5)

    # Apply bilateral filtering
    image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)
    for i in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)
    for i in range(3):
        image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)

    # Sharpen the image
    gaussian_mask = cv2.GaussianBlur(image_filtered, (7, 7), 2)
    image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
    image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)

    return image_sharp

# Route for home page
@app.route('/')
def home():
    return render_template('index-1.html')

# Route to process the image
@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    if file:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_image = process_image(image)
        
        _, buffer = cv2.imencode('.jpg', processed_image)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

# Run the app
if __name__ == "__main__":
    # Debugging print statements to check if Flask can find templates
    print("Current working directory:", os.getcwd())
    print("Templates folder exists:", os.path.exists("templates"))
    print("index.html exists:", os.path.exists("templates/index.html"))

    app.run(debug=True)
