# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

from . import craft_utils
from . import imgproc
from . import file_utils

from .craft import CRAFT

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

def add_margin(poly, margin):
  """
  Function to apply margin to open cv polyfill.
  Assumes poly in the format of an array with shape (4,2).
  Assumes endpoint will be given in the order of (left_top, right_top, right_bottom, left_bottom)
  """
  poly_new = poly.copy()
  poly_new = np.array([[poly[0][0] - margin, poly[0][1] - margin],
                       [poly[1][0] + margin, poly[1][1] - margin],
                       [poly[2][0] + margin, poly[2][1] + margin],
                       [poly[3][0] - margin, poly[3][1] + margin]
                       ])
  return poly_new

def does_overlap(poly_1, poly_2):
  """
  Returns boolean, True if the two poly endpoints touch
  """
  #Check if all of poly_1 is left of poly_2
  if poly_1[0][0] > poly_2[1][0] and poly_1[3][0] > poly_2[2][0]:
    return False

  #Check if all of poly_1 is right of poly_2
  if poly_1[1][0] < poly_2[0][0] and poly_1[2][0] < poly_2[3][0]:
    return False

  #Check if all of poly_1 is above poly_2
  if poly_1[3][1] < poly_2[0][1] and poly_1[2][1] < poly_2[1][1]:
    return False

  #Check if all of poly_1 is below poly_2
  if poly_1[0][1] > poly_2[3][1] and poly_1[1][1] > poly_2[2][1]:
    return False

  return True

def merge_bboxes(poly_1, poly_2):
  """
  Merge two bounding boxes
  """
  return np.array([[min(poly_1[0][0], poly_2[0][0]), min(poly_1[0][1], poly_2[0][1])],
                   [max(poly_1[1][0], poly_2[1][0]), min(poly_1[1][1], poly_2[1][1])],
                   [max(poly_1[2][0], poly_2[2][0]), max(poly_1[2][1], poly_2[2][1])],
                   [min(poly_1[3][0], poly_2[3][0]), max(poly_1[3][1], poly_2[3][1])]
                  ])

def _mask_merge_text_regions(bboxes, margin):
  """
  Merge mask poly shapes that are within margin distance of each other
  """
  mask_bboxes = list(bboxes.copy())

  for i in range(len(mask_bboxes)):
    mask_bboxes[i] = add_margin(mask_bboxes[i], margin)

  #Merge boxes
  merging = True
  while merging:
    merging = False
    for i, box_1 in enumerate(mask_bboxes):
      for j, box_2 in enumerate(mask_bboxes):
        if i != j:
          if does_overlap(box_1, box_2):
            merging = True
            mask_bboxes[i] = merge_bboxes(box_1, box_2)
            mask_bboxes.pop(j)
            break

  return mask_bboxes

def CRAFT_inference(craft_model, image, text_threshold=0.4, link_threshold=0.4, low_text=0.4, cuda=True, poly=False, refine_text=None, canvas_size=1280):
  t0=time.time()

  #resize
  img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
  ratio_h = ratio_w = 1 / target_ratio

  #preprocessing
  x = imgproc.normalizeMeanVariance(img_resized)
  x = torch.from_numpy(x).permute(2,0,1) #[h,w,c] to [c,h,w]
  x = Variable(x.unsqueeze(0)) #[c,h,w] to [b,c,h,w]
  if cuda:
    x = x.cuda()

  #forward pass
  with torch.no_grad():
    y, feature = craft_model(x)

  #make score and link map
  score_text = y[0,:,:,0].cpu().data.numpy()
  score_link = y[0,:,:,1].cpu().data.numpy()

  #refine link
  t0 = time.time() - t0
  t1 = time.time()

  #post processing
  boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
  # coordinate adjustment
  boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
  polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

  for k in range(len(polys)):
    if polys[k] is None: polys[k] = boxes[k]

  return boxes, polys, score_text

def blur_and_save(img_file, img, boxes, margin, dirname="./blurred_images/"):
  img = np.array(img)

  #make result file list
  filename, file_ext = os.path.splitext(os.path.basename(img_file))

  # result directory
  blurred_img_file = dirname + filename + ".png" #Using png instead of jpg

  if not os.path.isdir(dirname):
    os.mkdir(dirname)

  #Initiate mask
  mask = np.zeros_like(img)
  start_point = (1050, 1600)
  end_point = (1600, 0)
  mask = cv2.rectangle(mask, start_point, end_point, (255, 255, 255), -1) #blur side bar thing?
  m_boxes = _mask_merge_text_regions(boxes, margin)
  for i, box in enumerate(m_boxes):
    poly = np.array(box).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2) We replace this with mask generation :)
    cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], color=(255, 255, 255))
    pt_color = (0,255,255)
    blurred_img_1000 = cv2.GaussianBlur(img, (51, 51), 1000)
    out = np.where(mask!=(255, 255, 255), img, blurred_img_1000)

  #save result
  cv2.imwrite(blurred_img_file, out)

def CRAFT_Blur_directory(model, directory, margin=12, cuda=True, res_dirname="./blurred_images/"):

  #generate image list from directory passed
  image_list, _, _ = file_utils.get_files(directory)

  #check if results folder exists
  results_folder = res_dirname
  if not os.path.isdir(results_folder):
      os.mkdir(results_folder)

  #Loading model and performing inference
  craft_model = CRAFT()
  craft_model.load_state_dict(copyStateDict(torch.load(model)))

  if cuda:
    craft_model = craft_model.cuda()
    craft_model = torch.nn.DataParallel(craft_model)
    cudnn.benchmark = False

  craft_model.eval()
  t = time.time()

  for k, image_path in enumerate(image_list): #load data
    image = imgproc.loadImage(image_path)
    bboxes, polys, score_text = CRAFT_inference(craft_model, image)

    blur_and_save(image_path, image[:,:,::-1], polys, margin, res_dirname)

  print("elapsed time : {}s".format(time.time() - t))


