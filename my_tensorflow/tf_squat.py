# Import TF and TF Hub libraries.
import tensorflow as tf
import os
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly() 
print(tf.executing_eagerly()) 
from thunder_pose_locations import landmarks

print(landmarks)

#Download the model from TF Hub.
#model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
model = tf.saved_model.load('/home/jacob_chiachuyou/')
movenet = model.signatures['serving_default']

image_paths = ['/home/jacob_chiachuyou/my_tensorflow/test']

directory = os.fsencode(image_paths[0])

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpg"): 
        print(os.path.join(directory.decode("utf-8"), filename))

#list of constants to deal with
#for calculating shoulder-hip (rounding of the back)
min_shoulder_hip_distance = 0
max_shoulder_hip_distance = 0

min_ear_shoulder_distance = 0
max_ear_shoulder_distance = 0

min_hip_y = 0
max_hip_y = 0

body_height = 0

#for the bar path
min_wrist_x = 0
max_wrist_x = 0

#for the feet movement tracking
feet_min_x = 0
feet_max_x = 0
feet_min_y = 0
feet_max_y = 0

is_full_squat = False
directory = os.fsencode(image_paths[0])
count = 0
report = {}

for file in sorted(os.listdir(directory)):
    count += 10
    filename = os.fsdecode(file)
    image_path = os.path.join(directory.decode("utf-8"), filename)
    print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
   # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    print(keypoints)
    outputs_numpy = outputs['output_0'].numpy()

    #determine if the right or left side should be used based on the average confidence score
    total_left_confidence = 0
    total_right_confidence = 0
    for i in range(17):
        if outputs_numpy[0][0][i][2] <= 0.3:
            continue
        if i %2 == 0 and i != 0:
            total_right_confidence += outputs_numpy[0][0][i][2]
            if i == 16: 
                print(total_right_confidence)
                print(total_left_confidence)
        elif i%2 == 1:
            total_left_confidence += outputs_numpy[0][0][i][2]
    #deal with the correct side
    dominant_side = ''
    if total_left_confidence > total_right_confidence:
        dominant_side = 'left'
    else:
        dominant_side = 'right'

    #upper back based on ear to shoulder distance
    ear_shoulder_dist =  abs(outputs_numpy[0][0][landmarks[dominant_side + ' ear']][1] -  outputs_numpy[0][0][landmarks[dominant_side + ' shoulder']][1])
    if ear_shoulder_dist >= 0.05:
        if report.get(str(count)) == None:
            report[str(count)] = ['Problem Spotted: rounded upper back, lookout!']
        else:
            report.get(str(count)).append('Problem Spotted: rounded upper back, lookout!')
        print('Problem Spotted: rounded upper back, lookout!')



    #handle lowerback distance computation
    shoulder_hip_dist =  outputs_numpy[0][0][landmarks[dominant_side + ' shoulder']][1] -  outputs_numpy[0][0][landmarks[dominant_side + ' hip']][1]
    if min_shoulder_hip_distance ==  0 and max_shoulder_hip_distance == 0:
        min_shoulder_hip_distance = shoulder_hip_dist
        max_shoulder_hip_distance = shoulder_hip_dist
    else:
        min_shoulder_hip_distance = min(shoulder_hip_dist, min_shoulder_hip_distance)
        max_shoulder_hip_distance = max(shoulder_hip_dist, max_shoulder_hip_distance)
    if  max_shoulder_hip_distance -  min_shoulder_hip_distance  >= 0.2:
        print('Problem Spotted: rounded lower back, lookout!')

    hip_y = outputs_numpy[0][0][landmarks[dominant_side + ' hip']][0]
    knee_y = outputs_numpy[0][0][landmarks[dominant_side + ' knee']][0]
    
    hip_knee_y_dist = hip_y - knee_y
    if hip_knee_y_dist <= 0.1:
        is_full_squat = True
    elif hip_knee_y_dist < 0:
        print("Potential over squat!")
    
    #feet should not move more than a certain distance
    feet_curr_x = outputs_numpy[0][0][landmarks[dominant_side + ' ankle']][1]
    feet_max_x = max(feet_max_x, feet_curr_x)
    feet_min_x = min(feet_min_x if feet_min_x else 1, feet_curr_x)


    feet_curr_y = outputs_numpy[0][0][landmarks[dominant_side + ' ankle']][0]
    feet_max_y = max(feet_max_y, feet_curr_y)
    feet_min_y = min(feet_min_y if feet_min_y else 1,  feet_curr_y)

    if feet_max_x - feet_min_x >= 0.1 or feet_max_y - feet_min_y >= 0.1:
        print("min_x : " + str(feet_min_x))
        print("max_x : " + str(feet_max_x))
        print("min_y : " + str(feet_min_y))
        print("may_y : " + str(feet_max_y))
        print('Problem Spotted: feet should not be moving around during the process of lifting!')

    #compare the knees over toes problem - based on relative position of feet to knee
    knee_x = outputs_numpy[0][0][landmarks[dominant_side + ' knee']][1]
    if abs(feet_curr_x - knee_x) >= 0.1:
        print("knees are too far foward of toes, danger of overextending knee joint!")

if not is_full_squat:
    print('did not make a full squat')

print(dominant_side)
print(report)
