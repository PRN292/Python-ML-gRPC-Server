import cv2
import base64
import requests
import json
import time

RESIZE_FACTOR = 2
FRAME_PER_PROCESS = 4


def process_frame(frame):
    # preprocess images
    small_frame = cv2.resize(frame, (0, 0), fx=1 / RESIZE_FACTOR, fy=1 / RESIZE_FACTOR)
    localhost = "http://127.0.0.1:5000/api/v1/processImage"
    success, code = cv2.imencode('.png', small_frame)
    base64_string = str(base64.b64encode(code))[2:-1]
    data = {
        'image': base64_string
    }
    x = requests.post(localhost, json=json.dumps(data))
    response_data = json.loads(x.text)
    print(response_data)
    return response_data


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


def main():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    i = 0
    result = None
    while True:
        ret, frame = video_capture.read()
        if i % FRAME_PER_PROCESS == 0:
            start = time.time()
            result = process_frame(frame)
            end = time.time()
            print('Time: ' + str(end-start))
        i += 1
        frame = draw_frame(frame, result)
        # Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the web cam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
