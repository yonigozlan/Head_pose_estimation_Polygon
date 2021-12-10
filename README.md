# Head_pose_estimation_Polygon
In this repository, you will find the code I used to do the who's who exercise along with the two SOTA implementation I used to get head pose estimations on the given samples.

## Setup
The two SOTA implementations don't use the same version of python. I advise to use different virtual environments for each of them.
WHENet uses python 3.6.13, and 3DDFA_V2 python 3.8.0.
Each implementation has its own requirements.txt to install.

### WHENet
In order to use WHENet, you must download YOLOv3 model for head detection and put it under WHENet/yolo_v3/data using this link:
https://drive.google.com/file/d/1wGrwu_5etcpuu_sLIXl9Nu0dwNc8YXIH/view?usp=sharing.

### 3DDFA_V2
This implementation uses Cpython and must therefore be built before being used.
place yourself in the 3DDFA_V2 folder and run:
```
sh ./build.sh
```

## Usage
To test an implementation on a video file, place yourself in the folder of the implementation you want to test (WHENet or 3DDFA_V2), and run:
```
python main_"implementation".py --video "path_to_video_file" --output "path_to_output_file"
```
Example:
In 3DDFA_V2 folder, run:
```
python main_3DDFA_V2.py --video ../samples/sample-0.mp4 --output ../results/3DDFA_V2/sample-0-traced.avi
```

