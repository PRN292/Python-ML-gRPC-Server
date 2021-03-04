import flask
from flask import request, jsonify
import main.detect
import json
from main import firebase, image_utils, detect
import uuid

app = flask.Flask(__name__)
app.config['DEBUG'] = True
labels_path = "yolo-custom\\obj.names"
weights_path = "yolo-custom\\yolov4-tiny-custom_final_mask.weights"
config_path = "yolo-custom\\yolov4-tiny-custom-mask.cfg"
detect = detect.FaceRecognition(weights_path, config_path, labels_path)
firebase_object = firebase.FireBase('../strangerdetection-firebase-adminsdk-ndswy-371433d43f.json',
                                    'strangerdetection.appspot.com')

'''
accept json with format {image: image_base64_string}
return json with format [{
                'face_location': face_location,
                'match': match
                }, 
                {
                'face_location': face_location,
                'match': match
                }, ....
                ]   
'''
@app.route('/api/v1/processImage', methods=['POST'])
def process_image():
    try:
        data = json.loads(request.get_json())
        if 'image' in data:
            image_base64_string = data['image']
            image_array = main.image_utils.parse_b64_string_to_image_array(image_base64_string)
            result = detect.process_image(image_array)
            return jsonify(result)
        else:
            return jsonify({'error': 'JSON wrong format'})
    except Exception as e:
        return jsonify({'error': str(e)})


'''
accept json with format {user_email: tient@gmail.com, image: image_base64_string}
return json with format {'encoding': encoding.tolist(),
                             'user_email': user_email,
                             'image_name': image_name}
'''
@app.route("/api/v1/encodings", methods=['POST'])
def create_encoding():
    try:
        data = json.loads(request.get_json())
        if 'image' in data and 'user_email' in data:
            user_email = data['user_email']
            image_base64_string = data['image']
            image_array = image_utils.parse_b64_string_to_image_array(image_base64_string)
            encoding = detect.get_encoding(image_array)
            image_name = str(uuid.uuid4()) + '.png'
            encoding_data = {'encoding': encoding.tolist(),
                             'user_email': user_email,
                             'image_name': image_name}
            firebase_object.upload_image(image_name, image_base64_string)
            firebase_object.create_encoding(encoding_data)
            detect.add_encoding(encoding)
            return jsonify(encoding_data)
        else:
            return jsonify({'error': 'JSON wrong format'})
    except Exception as e:
        return jsonify({'error': e})


'''
accept json with format {'image_name': 'image_name'}
return json with format {'count': count}
'''
@app.route("/api/v1/encodings", methods=['DELETE'])
def delete_encoding():
    data = json.loads(request.get_json())
    try:
        if 'image_name' in data:
            image_name = data['image_name']
            encodings = firebase_object.delete_encoding_by_image_name(image_name)
            for encoding in encodings:
                detect.remove_encoding(encoding)
            return jsonify({'count': len(encodings)})
        else:
            return jsonify({'error': 'JSON wrong format'})
    except Exception as e:
        return jsonify({'error': e})


@app.route('/home')
def home():
    return 'Hello'


if __name__ == '__main__':
    detect.set_encodings(firebase_object.get_all_encoding_value())
    app.run()
