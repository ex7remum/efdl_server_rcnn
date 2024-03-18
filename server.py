import json

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import requests
from PIL import Image
import io


app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)

model = torchvision.models.detection.maskrcnn_resnet50_fpn()
model.load_state_dict(torch.load('maskrcnn_resnet50_fpn.pt'))
model.eval()
with open('labels.json', 'r') as f:
    labels_raw = json.loads(f.read())
    labels = {int(index): value for index, value in enumerate(labels_raw)}


@app.route("/predict", methods=['POST'])
@metrics.counter("app_http_inference_count", "number of invocations")
def predict():

    url = request.get_json(force=True)["url"]
    response = requests.get(url)
    data = Image.open(io.BytesIO(response.content))
    tensor_data = transforms.ToTensor(data).unsqueeze(0)
    batch = convert_image_dtype(tensor_data, dtype=torch.float)

    with torch.no_grad():
        out = model(batch)

    cur_labels = out[0]['labels']
    scores = out[0]['scores']
    res_labels = []
    for label, score in zip(cur_labels, scores):
        if score > 0.75:
            res_labels.append(labels[label])

    return jsonify({
        "objects": res_labels
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
