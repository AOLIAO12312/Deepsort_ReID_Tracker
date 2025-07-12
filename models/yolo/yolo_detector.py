from types import SimpleNamespace

import numpy as np
import torch
import platform

if platform.system() != 'Windows':
    from detector.nms import nms_wrapper
from detector.yolo.bbox import bbox_iou
from detector.yolo.util import unique

"""API of yolo detector"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform
from alphapose.utils.config import update_config
from detector.yolo.darknet import Darknet
from detector.yolo.preprocess import prep_image, prep_frame

from ultralytics import YOLO

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
    # video='data/Camera1.mp4',
    webcam=-1,
    save_video=True,
    vis_fast=True,
    pose_flow=False,
    pose_track=False,
    device=torch.device(type='cuda', index=0),
    tracking=False
)

cfg = update_config(args.cfg)

class YoloDetector:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        # self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.detector_cfg = cfg
        self.detector_opt = args
        self.model_cfg = cfg.get('CONFIG', 'detector/yolo/cfg/yolov3-spp.cfg')
        self.model_weights = cfg.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights')
        self.inp_dim = cfg.get('INP_DIM', 608)
        self.nms_thres = cfg.get('NMS_THRES', 0.6)
        self.confidence = 0.3 if (False if not hasattr(args, 'tracking') else args.tracking) else cfg.get('CONFIDENCE',0.05)
        self.num_classes = cfg.get('NUM_CLASSES', 80)

        print('Loading YOLO model..')
        self.detector = YOLO(model_path).to(self.device)


        # self.model = Darknet(self.model_cfg)
        # self.model.load_weights(self.model_weights)
        # self.model.net_info['height'] = self.inp_dim
        # self.model.to(device)
        # self.model.eval()

    def get_result(self,frames):
        return self.detector.predict(frames,conf=0.01,       # 超低置信度阈值
                                            iou=0.3,         # 更宽松的NMS抑制
                                            classes=[0],     # 只检测 person
                                            agnostic_nms=True,  # 类别无关NMS（避免互相抑制）
                                            verbose=False)

    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)
        Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))
        """
        if isinstance(img_source, str):
            img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)
        elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):
            img, orig_img, im_dim_list = prep_frame(img_source, self.inp_dim)
        else:
            raise IOError('Unknown image source type: {}'.format(type(img_source)))

        return img

    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into object detection network and
        collect bbox w.r.t original image size
        Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size
        Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results
        """
        # 目标检测
        args = self.detector_opt
        _CUDA = True
        if args:
            if args.gpus[0] < 0:
                _CUDA = False
        with torch.no_grad():
            imgs = imgs.to(args.device) if args else imgs.cuda()
            prediction = self.model(imgs, args=args)  # torch.Size([5, 22743, 85])
            # do nms to the detection results, only human category is left
            dets = self.dynamic_write_results(prediction, self.confidence,
                                              self.num_classes, nms=True,
                                              nms_conf=self.nms_thres)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return 0
            dets = dets.cpu()

            orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
            dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)) / 2
            dets[:, 1:5] /= scaling_factor
            for i in range(dets.shape[0]):
                dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
                dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

            # 返回值 dets 的结构说明：
            # dets.shape = [num_detections, 8]
            # 每一行表示一个检测到的目标框（已完成 NMS 和坐标还原），字段含义如下：
            # [image_idx, x1, y1, x2, y2, objectness_score, class_score, class_index]

            # TODO：在此处插入目标筛选的代码
            return dets  # torch.Size([254, 8])

    def dynamic_write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        prediction_bak = prediction.clone()
        dets = self.write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            nms_conf -= 0.05
            dets = self.write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)

        return dets

    def write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        args = self.detector_opt
        # prediction: (batchsize, num of objects, (xc,yc,w,h,box confidence, 80 class scores))
        conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
        prediction = prediction * conf_mask

        try:
            ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
        except:
            return 0

        # the 3rd channel of prediction: (xc,yc,w,h)->(x1,y1,x2,y2)
        box_a = prediction.new(prediction.shape)
        box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_a[:, :, :4]

        batch_size = prediction.size(0)

        output = prediction.new(1, prediction.size(2) + 1)
        write = False
        num = 0
        for ind in range(batch_size):
            # select the image from the batch
            image_pred = prediction[ind]

            # Get the class having maximum score, and the index of that class
            # Get rid of num_classes softmax scores
            # Add the class index and the class score of class having maximum score
            max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            seq = (image_pred[:, :5], max_conf, max_conf_score)
            # image_pred:(n,(x1,y1,x2,y2,c,s,idx of cls))
            image_pred = torch.cat(seq, 1)

            # Get rid of the zero entries
            non_zero_ind = (torch.nonzero(image_pred[:, 4]))

            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

            # Get the various classes detected in the image
            try:
                img_classes = unique(image_pred_[:, -1])
            except:
                continue

            # WE will do NMS classwise
            # print(img_classes)
            for cls in img_classes:
                if cls != 0:
                    continue
                # get the detections with one particular class
                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                # sort the detections such that the entry with the maximum objectness
                # confidence is at the top
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
                idx = image_pred_class.size(0)

                # if nms has to be done
                if nms:
                    if platform.system() != 'Windows':
                        # We use faster rcnn implementation of nms (soft nms is optional)
                        nms_op = getattr(nms_wrapper, 'nms')
                        # nms_op input:(n,(x1,y1,x2,y2,c))
                        # nms_op output: input[inds,:], inds
                        _, inds = nms_op(image_pred_class[:, :5], nms_conf)

                        image_pred_class = image_pred_class[inds]
                    else:
                        # Perform non-maximum suppression
                        max_detections = []
                        while image_pred_class.size(0):
                            # Get detection with highest confidence and save as max detection
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            # Stop if we're at the last detection
                            if len(image_pred_class) == 1:
                                break
                            # Get the IOUs for all boxes with lower confidence
                            ious = bbox_iou(max_detections[-1], image_pred_class[1:], args)
                            # Remove detections with IoU >= NMS threshold
                            image_pred_class = image_pred_class[1:][ious < nms_conf]

                        image_pred_class = torch.cat(max_detections).data

                # Concatenate the batch_id of the image to the detection
                # this helps us identify which image does the detection correspond to
                # We use a linear straucture to hold ALL the detections from the batch
                # the batch_dim is flattened
                # batch is identified by extra batch column

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))
                num += 1

        if not num:
            return 0
        # output:(n,(batch_ind,x1,y1,x2,y2,c,s,idx of cls))
        return output