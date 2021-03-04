import base64
import numpy as np
import cv2


def parse_b64_string_to_image_array(base64_string):
    b64 = base64.b64decode(base64_string)
    a = np.asarray(bytearray(b64), dtype='uint8')
    image = cv2.imdecode(a, cv2.IMREAD_COLOR)
    return image


