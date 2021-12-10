from copy import deepcopy
from PIL import Image

import pandas as pd
import cv2
import numpy as np


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

if __name__ == "__main__":
    '''
    0 -> c
    1 -> a
    2 -> b
    '''
    series = pd.read_parquet("series.pq")
    VIDEO_SRC = "../samples/sample-0.mp4"
    VECTOR_SET = 'c'
    VIDEO_OUTPUT = "./results/test.avi"
    cap = cv2.VideoCapture(VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # write the result to a video

    filter_col = [col for col in series if col.startswith(VECTOR_SET)]
    series = series[filter_col]
    series.rename(columns=lambda x: x[2:], inplace=True)
    for frame_nb in range(299):
        vector_dict = series.iloc[frame_nb].to_dict()
        print(frame_nb)
        ret, frame = cap.read()
        frame_traced = draw_head_pose(vector_dict, frame)
        img_pil = Image.fromarray(frame_traced)
        out.write(frame_traced)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()