import json
import logging

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype

from concurrent import futures

import grpc
import numpy as np
import torch
import inference_pb2
import inference_pb2_grpc

import requests
from PIL import Image
import io


class InstanceDetectorServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        with open('labels.json', 'r') as f:
            labels_raw = json.loads(f.read())
            self.labels = {int(index): value for index, value in enumerate(labels_raw)}

    def Predict(self, request, context):
        response = requests.get(request.url)
        data = Image.open(io.BytesIO(response.content))
        transform = transforms.ToTensor()
        tensor_data = transform(data).unsqueeze(0)
        batch = convert_image_dtype(tensor_data, dtype=torch.float)

        with torch.no_grad():
            out = self.model(batch)

        cur_labels = out[0]['labels']
        scores = out[0]['scores']
        res_labels = []
        for label, score in zip(cur_labels, scores):
            if score > 0.75:
                res_labels.append(self.labels[label.item()])
        return inference_pb2.InstanceDetectorOutput(objects=res_labels)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorServicer(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()