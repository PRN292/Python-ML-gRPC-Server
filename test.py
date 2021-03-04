import main.firebase
import requests
import base64
import json
import time

firebase = main.firebase.FireBase('../strangerdetection-firebase-adminsdk-ndswy-371433d43f.json',
                                  'strangerdetection.appspot.com')


def test_server():
    url = firebase.get_image_access_url("tien.png")
    content = requests.get(url).content
    localhost = "http://127.0.0.1:5000/api/v1/encodings"
    base64_string = str(base64.b64encode(content))[2:-1]

    data = {
        'user_email': 'tientt@gmail.com',
        'image': base64_string
    }
    print(json.dumps(data))
    x = requests.post(localhost, json=json.dumps(data))
    print(x.text)


def test_firebase():
    image_name = "71650dc8-3674-4702-a948-308ae0369a6a.png"
    print(firebase.get_image_access_url(image_name))


def test_create_new_encoding():
    localhost = "http://127.0.0.1:5000/api/v1/encodings"
    with open('../1440_Vladimir_Putin_Wallpaper.jpg', 'rb') as image:
        base64_string = str(base64.b64encode(image.read()))[2:-1]
    data = {
        'user_email': 'tientt@gmail.com',
        'image': base64_string
    }
    start = time.time()
    x = requests.post(localhost, json=json.dumps(data))
    end = time.time()
    print("Time: " + str(end - start))
    response_data = json.loads(x.text)
    print(response_data)
    image_name = response_data['image_name']
    url = firebase.get_image_access_url(image_name)
    print(url)


def test_delete_encoding():
    localhost = "http://127.0.0.1:5000/api/v1/encodings"
    data = {
        'image_name': '71d7c0e2-417d-4d27-90f3-51acc14126a3.png'
    }
    start = time.time()
    x = requests.delete(localhost, json=json.dumps(data))
    end = time.time()
    print("Time: " + str(end - start))
    response_data = json.loads(x.text)
    print(response_data)


def test_detect():
    localhost = "http://127.0.0.1:5000/api/v1/processImage"
    with open('../tien.png', 'rb') as image:
        base64_string = str(base64.b64encode(image.read()))[2:-1]
    data = {
        'image': base64_string
    }
    start = time.time()
    x = requests.post(localhost, json=json.dumps(data))
    end = time.time()
    print("Time: " + str(end - start))
    response_data = json.loads(x.text)
    print(response_data)


if __name__ == '__main__':
    #test_create_new_encoding()
    # test_delete_encoding()
    test_detect()
