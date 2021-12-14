'''
Code to generate videos with boxes corresponding to the parquet file for the who's who exercise
'''

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
    # Configurability
    # change VECTOR_SET to "a", "b" or "c" and the number in VIDEO_SRC to test each vector set on each video
    VIDEO_SRC = "../samples/sample-0.mp4"
    VECTOR_SET = 'c'
    VIDEO_OUTPUT = "./results/test.avi"
    # get only the vector set specified above
    filter_col = [col for col in series if col.startswith(VECTOR_SET)]
    series = series[filter_col]
    series.rename(columns=lambda x: x[2:], inplace=True)
    # setup the parameters to open, analyze the initial video, and save the modified video.
    cap = cv2.VideoCapture(VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # write the result to a video
    # "299" shouldn't be hardcoded but I couldn't get the number of frame in a video, and the 3 samples are of the same length
    for frame_nb in range(299):
        print(frame_nb)
        vector_dict = series.iloc[frame_nb].to_dict()
        ret, frame = cap.read()
        # draw the head pose on the current frame
        frame_traced = draw_head_pose(vector_dict, frame)
        img_pil = Image.fromarray(frame_traced)
        # write the modified frame to the output video
        out.write(frame_traced)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()