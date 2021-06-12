# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

from thunder_pose_locations import landmarks

print(landmarks)

# Download the model from TF Hub.
#model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
model = tf.saved_model.load('/home/jacob_chiachuyou/')
movenet = model.signatures['serving_default']

image_paths = []


#list of constants to deal with
#for calculating shoulder-hip (rounding of the back)
min_shoulder_hip_distance = 0
max_shoulder_hip_distance = 0

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


for image_path in image_paths:
    # Load the input image.
    image_path = '/home/jacob_chiachuyou/'+ image_path
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
    
    outputs_numpy = outputs.numpy()

    #determine if the right or left side should be used based on the average confidence score
    total_left_confidence = 0
    total_right_confidence = 0
    for i in range(17):
        if i %2 == 0:
            total_right_confidence += outputs_numpy[0][0][i][2]
        elif i != 0 and i%2 == 1:
            total_left_confidence += outputs_numpy[0][0][i][2]

    #deal with the correct side
    dominant_side = ''
    if total_left_confidence > total_right_confidence:
        dominant_side = 'left'
    else:
        dominant_side = 'right'


    #handle backarch distance computation
    shoulder_hip_dist =  outputs_numpy[0][0][landmarks[dominant-side + ' shoulder'][1] -  outputs_numpy[0][0][landmarks[dominant-side + ' hip'][1] 
    if min_shoulder_hip_distance ==  0 and max_shoulder_hip_distance == 0:
        min_shoulder_hip_distance = shoulder_hip_dist
        max_shoulder_hip_distance = shoulder_hip_dist
    else:
        min_shoulder_hip_distance = min(shoulder_hip_dist, min_shoulder_hip_distance)
        max_shoulder_hip_distance = max(,shoulder_hip_dist, max_shoulder_hip_distance)
    if  max_shoulder_hip_distance -  min_shoulder_hip_distance  >= 0.2:
        print('Problem Spotted: rounded back, lookout!')

    #handle hip parallel with shoulder and knee
    shoulder_knee = outputs_numpy[0][0][landmarks[dominant-side + ' shoulder'][0] -  outputs_numpy[0][0][landmarks[dominant-side + ' knee'][0] 
    shoulder_hip = outputs_numpy[0][0][landmarks[dominant-side + ' shoulder'][0] -  outputs_numpy[0][0][landmarks[dominant-side + ' hip'][0]
    hip_knee = outputs_numpy[0][0][landmarks[dominant-side + ' hip'][0] -  outputs_numpy[0][0][landmarks[dominant-side + ' knee'][0]
    if shoulder_knee > 0.01 and shoulder_hip > 0.01 and hip_knee > 0.01: 
        print('Problem Spotted: shoulders, hip and knee should be parallel!')

    wrist_curr_x = outputs_numpy[0][0][landmarks[dominant-side + ' wrist'][1]
    wrist_max_x = max(wrist_max_x, wrist_curr_x) 
    wrist_min_x = min(wrist_min_x, wrist_curr_x)
    #bar path to chest should be within a certain range
    if wrist_max_x - wrist_min_x >= 0.2:
        print('Problem Spotted: bar should not move sideways too far')

    #feet should not move more than a certain distance
    feet_curr_x = outputs_numpy[0][0][landmarks[dominant-side + ' ankle'][1]
    feet_max_x = max(feet_max_x, feet_curr_x) 
    feet_min_x = min(feet_min_x,  feet_curr_x)
    

    feet_curr_y = outputs_numpy[0][0][landmarks[dominant-side + ' ankle'][0]
    feet_max_y = max(feet_max_y, feet_curr_y) 
    feet_min_y = min(feet_min_y,  feet_curr_y)
     
    if feet_max_x - feet_min-x >= 0.05 or feet_max_y - feet_min_y >= 0.05:
        print('Problem Spotted: feet should not be moving around during the process of lifting!')
        
