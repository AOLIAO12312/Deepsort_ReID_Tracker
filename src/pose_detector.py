
from alphapose.models import builder # 装载姿态模型的核心包
from alphapose.utils.config import update_config # 读取配置文件

from types import SimpleNamespace
import torch
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from tqdm import tqdm
import numpy as np

from alphapose.models import builder # 装载姿态模型的核心包
from alphapose.utils.config import update_config # 读取配置文件

from types import SimpleNamespace
import torch
from alphapose.utils.detector import DetectionLoader
from detector.apis import get_detector
from alphapose.utils.transforms import get_func_heatmap_to_coord



# 定义args字面量
args = SimpleNamespace(
    cfg='configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml',
    checkpoint='pretrained_models/multi_domain_fast50_regression_256x192.pth',
    sp=True,
    detector='yolo',
    detfile='',
    inputpath='',
    inputlist='',
    inputimg='',
    outputpath='examples/res/',
    save_img=False,
    vis=False,
    showbox=False,
    profile=False,
    format=None,
    min_box_area=0,
    detbatch=5,
    posebatch=64,#一次性处理多少个batch的人物数据，由显存决定上限
    eval=False,
    gpus=[0],
    qsize=1024,
    flip=False,
    debug=False,
    video='data/Camera1.mp4',
    webcam=-1,
    save_video=True,
    vis_fast=True,
    pose_flow=False,
    pose_track=False,
    device=torch.device(type='cuda', index=0),
    tracking=False
)

checkpoint = args.checkpoint
cfg = update_config(args.cfg)

hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

# 将hm_data转换为坐标数据的核心函数
heatmap_to_coord = get_func_heatmap_to_coord(cfg)
norm_type = cfg.LOSS.get('NORM_TYPE', None)
class PoseDetector:
    def __init__(self):
        # Load pose model
        self.cfg = cfg
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.device = args.device
        print('Loading pose model from %s...' % (checkpoint,))
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        # 配置模型为推理模式
        self.pose_model.to(args.device)
        self.pose_model.eval()

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self.batchSize = args.posebatch

        self._sigma = cfg.DATA_PRESET.SIGMA
        if cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False,
                loss_type=cfg.LOSS['TYPE'])

        self.eval_joints = []

    def image_preprocess(self, frames: list):
        """
        将 frames 中的裁剪图像转换为 inps 张量和 cropped_boxes。
        依赖 self.transformation.test_transform。

        :param frames: List[np.ndarray] 裁剪后的人物图像列表
        :return: inps: torch.Tensor (N, 3, H, W)
                 cropped_boxes: torch.Tensor (N, 4)
        """
        inps = []
        cropped_boxes = []

        for frame in frames:
            if frame is None:
                continue
            h, w = frame.shape[:2]
            box = [0, 0, w, h]  # 使用整张图像作为裁剪框
            inp, cropped_box = self.transformation.test_transform(frame, box)
            inps.append(inp)
            cropped_boxes.append(torch.FloatTensor(cropped_box))

        if not inps:
            return torch.empty((0, 3, *self._input_size)), torch.empty((0, 4))

        inps_tensor = torch.stack(inps)
        cropped_boxes_tensor = torch.stack(cropped_boxes)

        return inps_tensor, cropped_boxes_tensor

    def get_result(self, frames:list):
        # frames 为裁切下的人物图像列表
        inps,cropped_boxes_tensor = self.image_preprocess(frames)
        with torch.no_grad():
            # Pose Estimation
            inps = inps.to(args.device)  # inps为张量，将其移动到GPU上
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % self.batchSize:
                leftover = 1
            num_batches = datalen // self.batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * self.batchSize:min((j + 1) * self.batchSize, datalen)]
                hm_j = self.pose_model(inps_j)  # 执行姿态检测模型推理
                hm.append(hm_j)
            hm = torch.cat(hm)
            hm = hm.cpu()
            # writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name) # 将hm数据送入writer进行处理

            # 解析hm数据
            hm_data = hm
            assert hm_data.dim() == 4

            face_hand_num = 110
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0, 136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0, 26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0, 133)]
            elif hm_data.size()[1] == 68:
                face_hand_num = 42
                self.eval_joints = [*range(0, 68)]
            elif hm_data.size()[1] == 21:
                self.eval_joints = [*range(0, 21)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes_tensor[i].tolist()
                if isinstance(heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                              norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            _result = []
            for k in range(datalen):
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        # 'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        # 'idx': ids[k],
                        # 'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                    }
                )
            return _result