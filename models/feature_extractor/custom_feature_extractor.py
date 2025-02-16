import torch
from torchreid.utils import FeatureExtractor
from src.utils import normalize_feature
class CustomFeatureExtractor:
    def __init__(self,extractor_name,extractor_path,device):
        self.extractor_path = extractor_path
        self.extractor_name = extractor_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.extractor = FeatureExtractor(
                    model_name=self.extractor_name,
                    model_path=self.extractor_path,
                    device=self.device,
                    verbose=False
                )

    def get_result(self,images):
        features = self.extractor(images)
        return features

    def get_normalized_result(self,images):
        normalized_features = normalize_feature(self.get_result(images))
        return normalized_features

    def get_extractor_name(self):
        return self.extractor_name