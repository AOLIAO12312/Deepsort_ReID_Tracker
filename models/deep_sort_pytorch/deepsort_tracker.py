from models.deep_sort_pytorch.utils.parser import get_config
from models.deep_sort_pytorch.deep_sort import DeepSort
import torch
class DeepsortTracker:
    def __init__(self,cfg_path:str,device:str = 'cuda:0'):
        self.cfg_path = cfg_path
        self.cfg = get_config()
        self.cfg.merge_from_file(cfg_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tracker = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                             max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                             nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                             max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                             max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                             use_cuda = False if self.device == 'cpu' else True)

    def get_cfg(self):
        return self.cfg

    def get_cfg_path(self):
        return self.cfg_path

    def get_tracker(self):
        return self.tracker