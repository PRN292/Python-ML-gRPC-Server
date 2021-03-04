from firebase_admin import credentials, firestore, initialize_app, storage
import datetime
import urllib
import numpy as np
import cv2
import uuid
import base64


class FireBase:
    def __init__(self, credential_file_path, storage_url):
        cred = credentials.Certificate(credential_file_path)
        default_app = initialize_app(cred, {
            'storageBucket': storage_url
        })
        self.db = firestore.client()
        self.bucket = storage.bucket(app=default_app)

    def get_image_access_url(self, image_name):
        blob = self.bucket.blob(image_name)
        return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

    @staticmethod
    def read_image_from_url(url):
        with urllib.request.urlopen(url) as response:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image

    def get_image_as_array(self, image_name):
        url = self.get_image_access_url(image_name)
        return self.read_image_from_url(url)

    def create_encoding(self, data: dict):
        encodings_collection = self.db.collection('encodings')
        encoding_id = str(uuid.uuid4())
        encodings_collection.add(data, encoding_id)

    def upload_image(self, image_name, image_base64_string):
        blob = self.bucket.blob(image_name)
        b64 = base64.b64decode(image_base64_string)
        blob.upload_from_string(b64, content_type='image/png')

    def delete_encoding_by_image_name(self, image_name):
        encodings_collection = self.db.collection('encodings')
        encodings_stream = encodings_collection.where(u'image_name', u'==', image_name).stream()
        encodings = []
        for encoding in encodings_stream:
            encodings_collection.document(encoding.id).delete()
            encodings.append(encoding.to_dict()['encoding'])
        return encodings

    def get_all_encoding_value(self):
        encodings_collection = self.db.collection('encodings')
        encodings_stream = encodings_collection.stream()
        encoding_values = []
        for encoding in encodings_stream:
            encoding_value = encoding.to_dict()['encoding']
            encoding_values.append(encoding_value)
        return encoding_values



