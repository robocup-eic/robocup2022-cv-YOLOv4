import argparse
import os
from pickle import NONE
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

#path
CONFIG_PATH = 'cfg/'
WEIGHTS_PATH = CONFIG_PATH + 'yolov4.weights'
NAMES_PATH = CONFIG_PATH + 'custom.names'
DEVICE = "" #'gpu' for socket and "" for colab
CFG_PATH = CONFIG_PATH + 'yolov4.cfg'
IMAGE_SIZE = 416

@torch.no_grad()

class ObjectDetection:
    
    def __init__(self):

        # Initialize
        self.device = select_device(DEVICE)
        # half precision only supported on CUDA
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # .cuda() #if you want cuda remove the comment
        self.model = Darknet(CFG_PATH, IMAGE_SIZE).cuda()
        try:
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device)['model'])
            #model = attempt_load(weights, map_location=device)  # load FP32 model
            #IMAGE_SIZE = check_img_size(IMAGE_SIZE, s=model.stride.max())  # check img_size
        except:
            load_darknet_weights(self.model, WEIGHTS_PATH)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()  # to FP16
            
        # Get names and colors
        self.names = self.load_classes(NAMES_PATH)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
    
    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def detect(self,input_image):

        #preprocess image
        input_image = self.preprocess(input_image)
        
        # Run inference
        t0 = time.time()
        # init img
        img = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)  
        # run once
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None 
        
        # Padded resize
        img = letterbox(input_image, new_shape=IMAGE_SIZE, auto_size=32)[0]

        # Convert 
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)
        
        print("recieving image with shape {}".format(img.shape))

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        print("Inferencing ...")
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres = 0.4, iou_thres =0.5, classes=None, agnostic=False)
        
        print("found {} object".format(len(pred)))

        # print string
        s= ''
        s += '%gx%g ' % img.shape[2:]  

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, input_image, label=label, color=self.colors[int(cls)], line_thickness=2)

        # Print time (inference + NMS)
        print('{}Done. {:.3} s'.format(s, time.time() - t0))
        
        return input_image
    
    def get_bbox(self, input_image):
        
        #preprocess image
        input_image = self.preprocess(input_image)
        
        # object bbox list
        bbox_list = []
        
        # Run inference
        t0 = time.time()
        # init img
        img = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)  
        # run once
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None 
        
        # Padded resize
        img = letterbox(input_image, new_shape=IMAGE_SIZE, auto_size=32)[0]

        # Convert 
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)
        
        print("recieving image with shape {}".format(img.shape))

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        print("Inferencing ...")
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres = 0.4, iou_thres =0.5, classes=None, agnostic=False)
        
        print("found {} object".format(len(pred)))

        # print string
        s= ''
        s += '%gx%g ' % img.shape[2:]  

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    temp = []
                    for ts in xyxy:
                        temp.append(ts.item())
                    bbox = list(np.array(temp).astype(int))
                    bbox.append(self.names[int(cls)])
                    bbox_list.append(bbox)

        # Print time (inference + NMS)
        print('{}Done. {:.3} s'.format(s, time.time() - t0))
        
        return bbox_list

    def preprocess(self, img):
        npimg = np.array(img)
        image = npimg.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
# def test():
#     OD = ObjectDetection()
#     with torch.no_grad() :
#       result_image = OD.detect('data\samples\bus.jpg')
#       result_box = OD.get_bbox('data\samples\bus.jpg')
#     print(result_image)
#     print(result_box)

# if __name__ == '__main__':
#     test()
