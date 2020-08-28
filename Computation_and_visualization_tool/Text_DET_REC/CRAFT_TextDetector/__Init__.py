"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from scipy.ndimage import rotate
from PIL import Image
import pytesseract

import cv2
from skimage import io
import numpy as np
import Computation_and_visualization_tool.Text_DET_REC.CRAFT_TextDetector.craft_utils
import Computation_and_visualization_tool.Text_DET_REC.CRAFT_TextDetector.imgproc
import Computation_and_visualization_tool.Text_DET_REC.CRAFT_TextDetector.file_utils
import json
import zipfile

from Computation_and_visualization_tool.Text_DET_REC.CRAFT_TextDetector.craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/CRAFT_TextDetector/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/CRAFT_TextDetector/figures/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


# """ For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

# result_folder = '/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/CRAFT_TextDetector/result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



# if __name__ == '__main__':
def detect_text(image):
    # load net
    net = CRAFT()     # initialize

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()


    # load data
    for root, dirs, files in os.walk("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/Deep_TextRecognition/Detected_Images/"):
        for file in files:
            os.remove(os.path.join(root, file))
    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
    box_centers=[]
    slope=[]
    l=[]
    dup_img = image.copy()
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        if(abs(poly[2][1]-poly[1][1])<100 and abs(abs(poly[2][1]-poly[1][1])-abs(poly[1][0]-poly[0][0]))<40):
            #if the item is single char
            slope += ['_']
        elif(abs(poly[1][0]-poly[0][0])>abs(poly[2][1]-poly[1][1])):
            slope += [(poly[1][1]-poly[0][1])/(poly[1][0]-poly[0][0])*45]
            l +=  [(poly[1][1]-poly[0][1])/(poly[1][0]-poly[0][0])*45]
        else:
            slope += [((poly[1][1]-poly[0][1])/(poly[1][0]-poly[0][0])*45)-90]
            l +=  [((poly[1][1]-poly[0][1])/(poly[1][0]-poly[0][0])*45)-90]
        x0,x1,y0,y1=(min(poly[:,0]),max(poly[:,0]),min(poly[:,1]),max(poly[:,1]))
        if x0<0:
            x0=0
        if y0<0:
            y0=0
        box_centers+=[[(x0+x1)/2,(y0+y1)/2]]

        ime_name="/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/Deep_TextRecognition/Detected_Images/"+str(i)+".png"
        Image.fromarray(image[y0:y1,x0:x1,:]).save(ime_name, 'PNG', dpi=(72,72))
        cv2.polylines(dup_img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    for i in range(len(bboxes)):
        if(slope[i]=='_'):
            slope[i] = 0
            if(len(l)!=0):
                slope[i]=np.mean(l)
        ime_name="/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_and_visualization_tool/Text_DET_REC/Deep_TextRecognition/Detected_Images/"+str(i)+".png"
        cv2.imwrite(ime_name,rotate(cv2.imread(ime_name,0), slope[i], cval=255))

    return dup_img,bboxes,box_centers,slope


    # for k, image_path in enumerate(image_list):
    #     print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
    #     image = imgproc.loadImage(image_path)
    #
    #     bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
    #     # # save score text
    #     # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    #     # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    #     # cv2.imwrite(mask_file, score_text)
    #     print(bboxes,"---",polys)
    #
    #     file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

