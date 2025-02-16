import torch
from ultralytics import YOLO
class YoloDetector:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.detector = YOLO(self.model_path).to(self.device)

    def get_model_path(self):
        return self.model_path

    def get_detector(self):
        return self.detector

    def get_result(self,images):
        return self.detector(images,verbose=False)[0]

