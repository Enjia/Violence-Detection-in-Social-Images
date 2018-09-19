import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
#import StringIO
import tensorflow as tf 
import numpy as np
import cv2
import time
# Obtain the flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER='img'
'''
def load_graph(trained_model):   
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph
'''
def load_sess_graph(export_dir):
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    print(sess, sess.graph)
    return sess, sess.graph

@app.route('/')
def index():
    return "Webserver is running"

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_size = 224
        num_channels = 3
        images = []
        # Reading the image using OpenCV
        image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, image_size,image_size,num_channels)

        start = time.time()
        sess = app.sess
        graph =app.graph

        input_x = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('fc/BiasAdd:0')
        output = tf.nn.sigmoid(output)
        #sess = tf.Session(graph=graph)
        out = sess.run(output, feed_dict={input_x: x_batch})
        end = time.time()
        result = {"protest": str(out[0][0]), "violence": str(out[0][1]), "sign": str(out[0][2]), "photo": str(out[0][3]), "fire": str(out[0][4]),
                  "police": str(out[0][5]), "children": str(out[0][6]), "group_20": str(out[0][7]), "group_100": str(out[0][8]), "flag": str(out[0][9]),
                  "night": str(out[0][10]), "shouting": str(out[0][11])}
        time_cost = end - start

        return jsonify(result, time_cost)

    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Running my first AI Demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">
          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">Violence Detection Demo</h1>
		            <h4 style="color:black">Upload new Image </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>
    '''


app.sess, app.graph = load_sess_graph('./tf_resnet50')  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
