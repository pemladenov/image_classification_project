import io
import numpy as np
import onnxruntime
import onnx
from flask_bootstrap import Bootstrap
from PIL import Image
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
Bootstrap(app)

inference = onnxruntime.InferenceSession('squeezenet1.0-12-int8.onnx', providers=['CPUExecutionProvider'])

# Load the model
model = onnx.load('squeezenet1.0-12-int8.onnx')

# Get the input node (the first node of the graph)
input_name = model.graph.input[0].name
print(input_name)


with open('static/synset.txt', 'r') as f:
    class_dict = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.astype(np.float32)
    # Assume img_arr has shape (height, width, channels)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension
    img_arr = np.transpose(img_arr, (0, 3, 1, 2))  # Transpose to (batch_size, channels, height, width)
    output = inference.run(None, {'data_0': img_arr})


    # make predictions
    output = inference.run(None, {'data_0': img_arr})
    probs = np.squeeze(output)

    # get the top 5 predicted classes and their probabilities
    top_indices = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_indices]
    top_classes = [class_dict[i] for i in top_indices]

    # return the result as a JSON object
    result = {'class': top_classes[0], 'probability': str(top_probs[0])}
    return jsonify(result)


    
if __name__ == '__main__':
    app.run()