import tensorflow as tf 
import numpy as np
import cv2
import time
import glob

export_dir = './tf_resnet50'
sess = tf.Session(graph=tf.Graph())
tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

files = glob.glob('/search/odin/Enjia/UCLA-protest/img/train'+'/*.jpg')
total_num = 1000
images = []

start = time.time()
for i in range(total_num):
	image = cv2.imread(files[i])
	image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)
	images.append(image)

images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)

x_batch = images.reshape(-1, 224, 224, 3)

graph = sess.graph
input_x = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('fc/BiasAdd:0')
output = tf.nn.sigmoid(output)
out = sess.run(output, feed_dict={input_x: x_batch})
end = time.time()
time = end - start
print("Total {} images classification costs {:.3} seconds, in average {:.3} seconds per image"
	.format(total_num, time, time/total_num))