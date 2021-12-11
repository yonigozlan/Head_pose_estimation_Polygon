import os
from copy import deepcopy
import argparse

import numpy as np
import cv2
from PIL import Image

import config
from src.yolo_v3.yolo_postprocess import YOLO
from src.run.predict import SADRNv2Predictor, Evaluator

def process_detection(predictor, evaluator, img, bbox , args ):

    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to make it squared
    y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max)
    if y_max-y_min > x_max-x_min:
        x_max_temp = x_max + (y_max-y_min)-(x_max-x_min)
        if x_max_temp > img.shape[1]:
            x_min -= (y_max-y_min)-(x_max-x_min)
        else :
            x_max = x_max_temp
    elif y_max-y_min < x_max-x_min:
        y_max_temp = y_max + (x_max-x_min) - (y_max-y_min)
        if y_max_temp > img.shape[0]:
            y_min -= (x_max-x_min) - (y_max-y_min)
        else :
            y_max = y_max_temp

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.resize(img_rgb, (256, 256))
    
    img_cropped = evaluator.evaluate_example_one_image(predictor, init_img=img_rgb, is_visualize=False)
    img_cropped = cv2.resize(img_cropped, (int(x_max-x_min), int(y_max-y_min)))
    
    img[y_min:y_max, x_min:x_max]= img_cropped
    return img


def main(args):
    yolo = YOLO(**vars(args))
    VIDEO_SRC = 0 if args.video == '' else args.video # if video clip is passed, use web cam
    cap = cv2.VideoCapture(VIDEO_SRC)
    print('cap info',VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video
    predictor_1 = SADRNv2Predictor(config.PRETAINED_MODEL)
    evaluator = Evaluator()

    while True:
        try:
            ret, frame = cap.read()
        except:
            break
        frame_rgb = frame
        img_pil = Image.fromarray(frame_rgb)
        bboxes, scores, classes = yolo.detect(img_pil)
        for bbox in bboxes:
            frame = process_detection(predictor_1, evaluator,frame, bbox, args)
        cv2.imshow('output', frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='whenet demo with yolo')
    parser.add_argument('--video', type=str, default='../samples/sample-0.mp4', help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', type=str, default='../results/SADRNet/sample-0-traced.avi', help='output video name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)