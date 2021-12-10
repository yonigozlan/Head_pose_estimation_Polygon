import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis
import os
import argparse
from yolo_v3.yolo_postprocess import YOLO
from PIL import Image
from copy import deepcopy


def deg_to_rad(angle):
    rad_angle = (np.pi / 180)*angle
    return rad_angle

def euler_to_rotation_vectors(yaw, pitch, roll):
    c1 = np.cos(yaw / 2)
    c2 = np.cos(pitch / 2)
    c3 = np.cos(roll / 2)
    s1 = np.sin(yaw / 2)
    s2 = np.sin(pitch / 2)
    s3 = np.sin(roll / 2)
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    return x, y, z

def process_detection( model, img, bbox, args ):

    y_min, x_min, y_max, x_max = bbox
    # enlarge the bbox to include more background margin
    # y_min = max(0, y_min - abs(y_min - y_max) / 10)
    # y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    # x_min = max(0, x_min - abs(x_min - x_max) / 5)
    # x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    # x_max = min(x_max, img.shape[1])

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )
    rotation_x, rotation_y, rotation_z = euler_to_rotation_vectors(deg_to_rad(yaw), deg_to_rad(pitch), deg_to_rad(roll))
    vector_dict = {
        "rotation-x": rotation_x,
        "rotation-y": rotation_y,
        "rotation-z": rotation_z,
        "translation-x": ((x_min+x_max)/2-img.shape[0]//2)//10,
        "translation-y": ((y_min+y_max)/2-img.shape[1]//2)//10,
        "translation-z": 600
    }
    print(vector_dict)
    # img = draw_head_pose(vector_dict, img)

    if args.display == 'full':
        cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return img


# mock_vector_structure = {
#   "rotation-x": 0.0,
#   "rotation-y": 0.0,
#   "rotation-z": 0.0,
#   "translation-x": 0.0,
#   "translation-y": 0.0,
#   "translation-z": 0.0
# }

def draw_head_pose(
    vector: dict, image: np.ndarray, color=(255, 255, 255), linewidth: float = 2
):
    if vector is None:
        return image

    # extract vectors from JSON snapshot
    rotation_vector = np.asarray(
        [vector.get(f"rotation-{key}") for key in ["x", "y", "z"]]
    ).reshape(-1, 1)
    translation_vector = np.asarray(
        [vector.get(f"translation-{key}") for key in ["x", "y", "z"]]
    ).reshape(-1, 1)

    # internal camera parameters
    image_focal = image.shape[1]
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    image_parameters = np.array(
        [
            [image_focal, 0, image_center[0]],
            [0, image_focal, image_center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )

    rear_size = 75
    rear_depth = 0
    front_size = 100
    front_depth = 100
    cloud_3D_model = []
    # build 3D facial model
    cloud_3D_model.append((-rear_size, -rear_size, rear_depth))
    cloud_3D_model.append((-rear_size, rear_size, rear_depth))
    cloud_3D_model.append((rear_size, rear_size, rear_depth))
    cloud_3D_model.append((rear_size, -rear_size, rear_depth))
    cloud_3D_model.append((-rear_size, -rear_size, rear_depth))
    cloud_3D_model.append((-front_size, -front_size, front_depth))
    cloud_3D_model.append((-front_size, front_size, front_depth))
    cloud_3D_model.append((front_size, front_size, front_depth))
    cloud_3D_model.append((front_size, -front_size, front_depth))
    cloud_3D_model.append((-front_size, -front_size, front_depth))
    cloud_3D_model = np.array(cloud_3D_model, dtype=float).reshape(-1, 3)

    # project 3D cloud to 2D plan
    (plan_2D_model, _) = cv2.projectPoints(
        cloud_3D_model,
        rotation_vector,
        translation_vector,
        image_parameters,
        np.zeros((4, 1)),
    )
    plan_2D_model = np.int32(np.round(plan_2D_model.reshape(-1, 2)))

    image_copy = deepcopy(image)
    # draw projected box
    cv2.polylines(image_copy, [plan_2D_model], True, color, linewidth, cv2.LINE_AA)
    cv2.line(
        image_copy,
        tuple(plan_2D_model[1]),
        tuple(plan_2D_model[6]),
        color,
        linewidth,
        cv2.LINE_AA,
    )
    cv2.line(
        image_copy,
        tuple(plan_2D_model[2]),
        tuple(plan_2D_model[7]),
        color,
        linewidth,
        cv2.LINE_AA,
    )
    cv2.line(
        image_copy,
        tuple(plan_2D_model[3]),
        tuple(plan_2D_model[8]),
        color,
        linewidth,
        cv2.LINE_AA,
    )

    return image_copy

def main(args):
    whenet = WHENet(snapshot=args.snapshot)
    yolo = YOLO(**vars(args))
    VIDEO_SRC = 0 if args.video == '' else args.video # if video clip is passed, use web cam
    cap = cv2.VideoCapture(VIDEO_SRC)
    print('cap info',VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video

    while True:
        try:
            ret, frame = cap.read()
        except:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        bboxes, scores, classes = yolo.detect(img_pil)
        for bbox in bboxes:
            frame = process_detection(whenet, frame, bbox, args)
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
    parser.add_argument('--video', type=str, default='IMG_0176.mp4',         help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', type=str, default='test.avi', help='output video name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)