import tensorflow as tf
from tensorflow.python.platform import gfile


LOGDIR='./logs'

with tf.Session() as sess:
    model_filename = 'mobilenet_quant_v1_224_classify_imagenet.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
