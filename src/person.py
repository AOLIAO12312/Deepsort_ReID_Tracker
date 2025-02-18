import torch
from collections import deque
import torch.nn.functional as F


class Person:
    def __init__(self, name: str, base_size=5, recent_size=20):
        self.name = name
        self.base_size = base_size
        self.recent_size = recent_size
        self.base_images = deque(maxlen=self.base_size)
        self.recent_images = deque(maxlen=self.recent_size)
        self.base_features = deque(maxlen=self.base_size)
        self.recent_features = deque(maxlen=self.recent_size)
        self.fused_feature = None

    def clear_recent_image_and_feature(self):
        self.recent_images.clear()
        self.recent_features.clear()

    def update_image_and_feature(self,image,feature,update_type:str):
        if image is not None and feature is not None and update_type is not None:
            ret0 = self.update_image(image,update_type)
            ret1 = self.update_feature(feature,update_type)
            if ret0 == True and ret1 == True:
                return True
        return False

    def update_image(self, image, image_type: str):
        if image_type == 'base':
            self.base_images.append(image)
        elif image_type == 'recent':
            self.recent_images.append(image)
        else:
            print(f"Image type can not be recognized")
            return False
        return True

    def update_feature(self, feature, feature_type: str):
        if feature_type == 'base':
            self.base_features.append(feature)
        elif feature_type == 'recent':
            self.recent_features.append(feature)
        else:
            print(f"Feature type can not be recognized")
            return False
        return True

    def get_all_images(self):
        return list(self.base_images) + list(self.recent_images)

    def get_all_features(self):
        return list(self.base_features) + list(self.recent_features)

    def get_name(self):
        return self.name

    def fuse_feature(self):
        all_features = self.get_all_features()
        if all_features:
            self.fused_feature = torch.mean(torch.stack(all_features), dim=0)
        else:
            print(f"No features available for fusion for {self.name}.")

    def get_fused_feature(self):
        if self.fused_feature is None:
            print(f"Fused feature for {self.name} is not computed yet.")
        return self.fused_feature

    def calculate_cosine_similarity(self, query_feature):
        similarity = F.cosine_similarity(query_feature.unsqueeze(0), self.fused_feature.unsqueeze(0))
        return similarity