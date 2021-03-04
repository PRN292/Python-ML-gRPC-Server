import grpc
import image_pb2_grpc
import image_pb2
from concurrent import futures
import cv2
from main import detect, image_utils, firebase
import base64
import uuid

FRAME_PER_PROCESS = 2
RESIZE_FACTOR = 4

labels_path = "D:\Programing\FaceRegconizing\main\yolo-custom\coco.names"
weights_path = "D:\Programing\FaceRegconizing\main\yolo-custom\yolov4-tiny-custom_final_mask.weights"
config_path = "D:\Programing\FaceRegconizing\main\yolo-custom\yolov4-tiny-custom-mask.cfg"
detect = detect.FaceRecognition(weights_path, config_path, labels_path)
firebase_object = firebase.FireBase(
    'D:\Programing\FaceRegconizing\strangerdetection-firebase-adminsdk-ndswy-371433d43f.json',
    'strangerdetection.appspot.com')

detect.set_encodings(firebase_object.get_all_encoding_value())


def process_frame(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=1 / RESIZE_FACTOR, fy=1 / RESIZE_FACTOR)
    result = detect.process_image(small_frame)
    return result


def draw_frame(frame, detect_info):
    for result in detect_info:
        (top, right, bottom, left) = result['face_location']
        match = result['match']
        # Scale back up face locations since the frame we detected
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = "Known" if match else "Unknown"
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame


class ProcessImageService(image_pb2_grpc.ProcessImageServicer):
    def ProcessImage(self, request, context):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Cannot open camera")
            exit()
        i = 0
        while True:
            ret, frame = video_capture.read()
            if i % FRAME_PER_PROCESS == 0:
                result = process_frame(frame)
            i += 1
            frame = draw_frame(frame, result)
            success, code = cv2.imencode('.png', frame)
            base64_string = str(base64.b64encode(code))[2:-1]
            yield image_pb2.ProcessImageReply(image=base64_string, result=str(result))

    def CreateEncoding(self, request, context):
        user_email = request.user_email
        image_base64_string = request.image
        image_array = image_utils.parse_b64_string_to_image_array(image_base64_string)
        encoding = detect.get_encoding(image_array)
        image_name = str(uuid.uuid4()) + '.png'
        encoding_data = {'encoding': encoding.tolist(),
                         'user_email': user_email,
                         'image_name': image_name}
        firebase_object.upload_image(image_name, image_array)
        firebase_object.create_encoding(encoding_data)
        return image_pb2.CreateEncodingReply(encoding=encoding, user_email=user_email, image_name=image_name)

    def DeleteEncoding(self, request, context):
        image_name = request.image_name
        encodings = firebase_object.delete_encoding_by_image_name(image_name)
        for encoding in encodings:
            detect.remove_encoding(encoding)
        return image_pb2.DeleteEncodingReply(count=len(encodings))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    server.add_insecure_port('[::]:50051')
    image_pb2_grpc.add_ProcessImageServicer_to_server(
        servicer=ProcessImageService(), server=server
    )
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    # detect.set_encodings(firebase_object.get_all_encoding_value())
    # print(firebase_object.get_image_access_url("214c9708-b8e3-4238-bde0-a9aaccb29d30.png"))
    serve()
