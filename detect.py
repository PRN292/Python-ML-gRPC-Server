import face_recognition
from main.yolo_detect import YoloFaceDetection


class FaceRecognition:
    def __init__(self, weights_path, config_path, labels_path):
        self.__encodings = []
        self.yolo_detect = YoloFaceDetection(weights_path, config_path, labels_path)

    def set_encodings(self, encodings: list):
        self.__encodings = encodings

    def add_encoding(self, encoding: list):
        self.__encodings.append(encoding)

    def remove_encoding(self, encoding: list):
        self.__encodings.remove(encoding)

    def get_encodings(self):
        return self.__encodings

    @staticmethod
    def detect_face(image_as_array):
        return face_recognition.face_locations(image_as_array)

    '''
        return [{
            'face_location': (x,y,a,b),
            'match': true/false
        }]
    '''

    def process_image(self, image_as_array):
        matches = []
        # find faces that match
        face_locations = self.yolo_detect.face_detect(image_as_array)
        # face_locations = self.detect_face(image_as_array)
        face_encodings = face_recognition.face_encodings(image_as_array, face_locations)
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(self.__encodings, face_encoding)
            # if face match one face in known face encodings
            matches.append(True in match)
        result = []
        for face_location, match in zip(face_locations, matches):
            result.append({
                'face_location': face_location,
                'match': match
            })
        return result, face_encodings

    @staticmethod
    def get_encoding(image_as_array):
        encodings = face_recognition.face_encodings(image_as_array)
        if len(encodings) != 1:
            raise Exception("Please choose image that has one face")
        return encodings[0]
