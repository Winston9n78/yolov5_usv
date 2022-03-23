from cgi import print_arguments
from csv import writer
from unittest import result
import cv2
import ffmpeg
import pylab
import imageio
import skimage
import numpy as np
import torch
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import letterbox


# img = cv2.imread("/home/ros/yolov5/data/images/1.jpg",1)
# print(img.shape)
#获得视频的格式
path = '/home/ros/yolov5/965.mp4'
videoCapture = cv2.VideoCapture(path)
videoCapture.open(0)
print(videoCapture.isOpened())
#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
 int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

#读帧
success, frame = videoCapture.read()

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

while success :
    print(1)
    # results = model(frame,  augment=False)
    # pred = non_max_suppression(results, 0.25, 0.45, None, False, max_det=1000)

    # for i, det in enumerate(pred):
    #     print(2)
    #     print(det)
    # cv2.imshow('windows', frame) #显示
    cv2.waitKey(int(1000/int(fps))) #延迟
    success, frame = videoCapture.read() #获取下一帧
    # if success:
    #     break

videoCapture.release()


#视频的绝对路径
#可以选择解码工具
# vid = imageio.get_reader(path, 'ffmpeg')
# fps = vid.get_meta_data()['fps']

# writer_ = imageio.get_writer('/home/ros/yolov5/966.mp4', fps = fps)

# # im = imageio.imread('/home/ros/yolov5/data/images/1.jpg')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

# for i, im in enumerate(vid):
#     # image = skimage.img_as_float(im).astype(np.float32)
#     # writer_.append_data(im[:, :, 1])
#     results = model(im)
#     # results = non_max_suppression(results, 0.25, 0.45, None, False, max_det=1000)
#     for i, im in enumerate(results):
#         print(1)

#     if i == 0:
#         break
# writer_.close()
# image 1/1: 720x1280 1 person, 2 boats
# Speed: 12.1ms pre-process, 190.2ms inference, 3.1ms NMS per image at shape (1, 3, 384, 640)