import os
import base64
from flask import Flask, request, jsonify
from ocr import load_ocr_model, process_image

app = Flask(__name__)

@app.route('/process_images', methods=['POST'])
def process_images():
    # Check if any files were uploaded
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'})

    files = request.files.getlist('files')

    # Check if any files were uploaded
    if len(files) == 0:
        return jsonify({'error': 'No files uploaded'})

    # Process each uploaded file
    results = {}
    ocr = load_ocr_model()

    for file in files:
        # Read the image file as binary data
        image_data = file.read()

        # Get the image name from the file name
        image_name = os.path.splitext(file.filename)[0]

        # Process the image data
        excel_output_path = process_image(image_name, image_data, ocr)

        # Store the output paths in the results dictionary
        results[image_name] = {'excel_output': excel_output_path}

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
