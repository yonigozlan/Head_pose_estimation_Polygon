# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml

import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.pose import viz_pose
from utils.functions import cv_draw_landmark, get_suffix


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']

    suffix = get_suffix(args.video_fp)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)

    # run
    dense_flag = args.opt in ('3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            # boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, ver, crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, pre_ver, crop_policy='landmark')

            roi_box = roi_box_lst
            # todo: add confidence threshold to judge the tracking is failed
            # if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
            boxes = face_boxes(frame_bgr)
            # boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

        pre_ver = ver  # for tracking

        if args.opt == '2d_sparse':
            res = cv_draw_landmark(frame_bgr, ver)
        elif args.opt == '3d':
            res = render(frame_bgr, [ver], tddfa.tri)
        elif args.opt == 'pose':
            res = viz_pose(np.array(frame_bgr), param_lst, ver)
        else:
            raise ValueError(f'Unknown opt {args.opt}')

        writer.append_data(res[..., ::-1])  # BGR->RGB

    writer.close()
    print(f'Dump to {video_wfp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '3d', 'pose'])
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)