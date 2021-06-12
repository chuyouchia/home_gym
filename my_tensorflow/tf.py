# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

# Download the model from TF Hub.
#model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
model = tf.saved_model.load('/home/jacob_chiachuyou/')
movenet = model.signatures['serving_default']

image_paths = ['pushup.jpeg', 'photo_2021-06-03_18-35-33.jpg', 'photo_2021-06-03_18-35-32.jpg','stock-photo-man-doing-chest-workout-bench-press-with-dumbbells-420962470.jpg']
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
