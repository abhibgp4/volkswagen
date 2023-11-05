from flask import Flask, render_template, request, jsonify
import os
import werkzeug
import testing_models_03
import cv2
import matplotlib.pyplot as plt

def imgshow():
    # Read your three images
    image1 = cv2.imread('templates/IMG/detected_face.jpg')
    image2 = cv2.imread('templates/IMG/predicted_image.png')  # Replace with the path to your third image

    # Create a figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the first image in the first subplot
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    ax1.set_title('Face Detected')

    # Display the second image in the second subplot
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    ax2.set_title('Model Prediction ')

    # Show the figure with all three subplots
    plt.show()

# Create a Flask web application
app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to render the index.html template
@app.route('/')
def hello_world():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the 'image' file is included in the POST request
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    image = request.files['image']  # Get the uploaded image

    # Check if the filename is empty (no file selected)
    if image.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    # Extract the original file extension from the filename
    file_extension = werkzeug.utils.secure_filename(image.filename).rsplit('.', 1)[1]

    # Save the uploaded image with the same filename and extension
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], f'test.{file_extension}'))

    # Call the process_image function from the emotion model module
    result = testing_models_03.process_and_display_images()
    imgshow()
    # Return the result from the ML model
    #return render_template('in2.html')
    return render_template('index.html')

    
if __name__ == '__main__':
    app.run(debug=True)

