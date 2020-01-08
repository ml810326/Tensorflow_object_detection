import paho.mqtt.client as mqtt
import time
import json
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class PredictFramework(object):
    def init(self):
        self.PATH_TO_CKPT = 'frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 1
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes)

                indices = np.argwhere(classes == 1)
                boxes = np.squeeze(boxes[indices])
                scores = np.squeeze(scores[indices])
                classes = np.squeeze(classes[indices])
                print(str(sum(inp > 0.5 for inp in scores)) + " person")
                if sum(inp > 0.5 for inp in scores) > 0:
                    return 1
                else:
                    return 0

class Runtime:

    def __init__(self, pf):

        MQTT_SERVER = "172.17.0.1"
        MQTT_PORT = 1883
        MQTT_ALIVE = 60
        self.MQTT_TOPIC = "viatest"
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(MQTT_SERVER, MQTT_PORT, MQTT_ALIVE)

        pf.init()

    def run(self):
        cap = cv2.VideoCapture(0)
        print("start to do")
        while True:
            ret, image = cap.read()
            resu = pf.detect(image)
            self.publish(resu)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    def publish(self, message):
        self.mqtt_client.publish(self.MQTT_TOPIC, json.dumps(message), qos=1)

if __name__ == '__main__':
    pf = PredictFramework()
    runtime = Runtime(pf)
    runtime.run()
