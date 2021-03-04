import cv2
import numpy as np


class YoloFaceDetection:
    def __init__(self, weights_path, config_path, labels_path):
        self.weights_path = weights_path
        self.config_path = config_path
        self.labels_path = labels_path
        self.net = cv2.dnn.readNet(config_path, weights_path)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    '''
        return indexes, boxes, classIDs, confidences
        indexes: numpy list position of which boxes shall be taken
        boxes: numpy list of boxes that YOLO can detect with 4 coordinates (top, left, width, height)
        classIDs: numpy list of classID corresponding with box
        confidence: numpy list of confidence corresponding with each box 

    '''
    def detect(self, image_as_array):
        h, w = image_as_array.shape[:2]
        blob = cv2.dnn.blobFromImage(image_as_array, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []
        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        return indexes, boxes, class_ids, confidences


    '''
        because yolo output coordinate is (top, left, width, height) but 
        face_recognizing library new coordinate (top, right, bottom, left)
        so we need to adjust it accordingly
    '''
    def output_info_for_framework(self, indexes, boxes):
        infos = []
        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                top, right, bottom, left = y, x + w, y + h, x
                infos.append((top, right, bottom, left))
        return infos

    def face_detect(self, image_as_array):
        indexes, boxes, class_ids, confidences = self.detect(image_as_array)
        return self.output_info_for_framework(indexes, boxes)
