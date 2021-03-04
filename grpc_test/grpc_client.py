import grpc
import image_pb2
import image_pb2_grpc
import cv2
from main import image_utils
import base64


def detect_image(stub):
    responses = stub.ProcessImage(image_pb2.ProcessImageRequest(id="123"))
    for response in responses:
        base64 = response.image
        frame = image_utils.parse_b64_string_to_image_array(base64)
        print(response.result)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def create_encoding(stub):
    user_email = "tientt@gmail.com"
    with open('../tien.png', 'rb') as image:
        base64_string = str(base64.b64encode(image.read()))[2:-1]
        print(base64_string)
        response = stub.CreateEncoding(
            image_pb2.CreateEncodingRequest(user_email=user_email, image=base64_string))
        print(response)


def delete_encoding(stub):
    image_name = 'e380819e-348d-436d-9c87-5407623c3877.png'
    response = stub.DeleteEncoding(image_pb2.DeleteEncodingRequest(image_name=image_name))
    print(response)


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = image_pb2_grpc.ProcessImageStub(channel)
        detect_image(stub)
        # create_encoding(stub)
        # delete_encoding(stub)


if __name__ == '__main__':
    run()
