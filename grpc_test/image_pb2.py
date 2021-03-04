# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: image.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='image.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0bimage.proto\"!\n\x13ProcessImageRequest\x12\n\n\x02id\x18\x01 \x01(\t\"2\n\x11ProcessImageReply\x12\r\n\x05image\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\t\":\n\x15\x43reateEncodingRequest\x12\x12\n\nuser_email\x18\x01 \x01(\t\x12\r\n\x05image\x18\x02 \x01(\t\"O\n\x13\x43reateEncodingReply\x12\x10\n\x08\x65ncoding\x18\x01 \x03(\x02\x12\x12\n\nuser_email\x18\x02 \x01(\t\x12\x12\n\nimage_name\x18\x03 \x01(\t\"+\n\x15\x44\x65leteEncodingRequest\x12\x12\n\nimage_name\x18\x01 \x01(\t\"$\n\x13\x44\x65leteEncodingReply\x12\r\n\x05\x63ount\x18\x01 \x01(\x05\x32\xca\x01\n\x0cProcessImage\x12:\n\x0cProcessImage\x12\x14.ProcessImageRequest\x1a\x12.ProcessImageReply0\x01\x12>\n\x0e\x43reateEncoding\x12\x16.CreateEncodingRequest\x1a\x14.CreateEncodingReply\x12>\n\x0e\x44\x65leteEncoding\x12\x16.DeleteEncodingRequest\x1a\x14.DeleteEncodingReplyb\x06proto3'
)




_PROCESSIMAGEREQUEST = _descriptor.Descriptor(
  name='ProcessImageRequest',
  full_name='ProcessImageRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ProcessImageRequest.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15,
  serialized_end=48,
)


_PROCESSIMAGEREPLY = _descriptor.Descriptor(
  name='ProcessImageReply',
  full_name='ProcessImageReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='ProcessImageReply.image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='result', full_name='ProcessImageReply.result', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=100,
)


_CREATEENCODINGREQUEST = _descriptor.Descriptor(
  name='CreateEncodingRequest',
  full_name='CreateEncodingRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='user_email', full_name='CreateEncodingRequest.user_email', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image', full_name='CreateEncodingRequest.image', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=102,
  serialized_end=160,
)


_CREATEENCODINGREPLY = _descriptor.Descriptor(
  name='CreateEncodingReply',
  full_name='CreateEncodingReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='encoding', full_name='CreateEncodingReply.encoding', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_email', full_name='CreateEncodingReply.user_email', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_name', full_name='CreateEncodingReply.image_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=162,
  serialized_end=241,
)


_DELETEENCODINGREQUEST = _descriptor.Descriptor(
  name='DeleteEncodingRequest',
  full_name='DeleteEncodingRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_name', full_name='DeleteEncodingRequest.image_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=243,
  serialized_end=286,
)


_DELETEENCODINGREPLY = _descriptor.Descriptor(
  name='DeleteEncodingReply',
  full_name='DeleteEncodingReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='count', full_name='DeleteEncodingReply.count', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=288,
  serialized_end=324,
)

DESCRIPTOR.message_types_by_name['ProcessImageRequest'] = _PROCESSIMAGEREQUEST
DESCRIPTOR.message_types_by_name['ProcessImageReply'] = _PROCESSIMAGEREPLY
DESCRIPTOR.message_types_by_name['CreateEncodingRequest'] = _CREATEENCODINGREQUEST
DESCRIPTOR.message_types_by_name['CreateEncodingReply'] = _CREATEENCODINGREPLY
DESCRIPTOR.message_types_by_name['DeleteEncodingRequest'] = _DELETEENCODINGREQUEST
DESCRIPTOR.message_types_by_name['DeleteEncodingReply'] = _DELETEENCODINGREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProcessImageRequest = _reflection.GeneratedProtocolMessageType('ProcessImageRequest', (_message.Message,), {
  'DESCRIPTOR' : _PROCESSIMAGEREQUEST,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:ProcessImageRequest)
  })
_sym_db.RegisterMessage(ProcessImageRequest)

ProcessImageReply = _reflection.GeneratedProtocolMessageType('ProcessImageReply', (_message.Message,), {
  'DESCRIPTOR' : _PROCESSIMAGEREPLY,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:ProcessImageReply)
  })
_sym_db.RegisterMessage(ProcessImageReply)

CreateEncodingRequest = _reflection.GeneratedProtocolMessageType('CreateEncodingRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATEENCODINGREQUEST,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:CreateEncodingRequest)
  })
_sym_db.RegisterMessage(CreateEncodingRequest)

CreateEncodingReply = _reflection.GeneratedProtocolMessageType('CreateEncodingReply', (_message.Message,), {
  'DESCRIPTOR' : _CREATEENCODINGREPLY,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:CreateEncodingReply)
  })
_sym_db.RegisterMessage(CreateEncodingReply)

DeleteEncodingRequest = _reflection.GeneratedProtocolMessageType('DeleteEncodingRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETEENCODINGREQUEST,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:DeleteEncodingRequest)
  })
_sym_db.RegisterMessage(DeleteEncodingRequest)

DeleteEncodingReply = _reflection.GeneratedProtocolMessageType('DeleteEncodingReply', (_message.Message,), {
  'DESCRIPTOR' : _DELETEENCODINGREPLY,
  '__module__' : 'image_pb2'
  # @@protoc_insertion_point(class_scope:DeleteEncodingReply)
  })
_sym_db.RegisterMessage(DeleteEncodingReply)



_PROCESSIMAGE = _descriptor.ServiceDescriptor(
  name='ProcessImage',
  full_name='ProcessImage',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=327,
  serialized_end=529,
  methods=[
  _descriptor.MethodDescriptor(
    name='ProcessImage',
    full_name='ProcessImage.ProcessImage',
    index=0,
    containing_service=None,
    input_type=_PROCESSIMAGEREQUEST,
    output_type=_PROCESSIMAGEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CreateEncoding',
    full_name='ProcessImage.CreateEncoding',
    index=1,
    containing_service=None,
    input_type=_CREATEENCODINGREQUEST,
    output_type=_CREATEENCODINGREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DeleteEncoding',
    full_name='ProcessImage.DeleteEncoding',
    index=2,
    containing_service=None,
    input_type=_DELETEENCODINGREQUEST,
    output_type=_DELETEENCODINGREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_PROCESSIMAGE)

DESCRIPTOR.services_by_name['ProcessImage'] = _PROCESSIMAGE

# @@protoc_insertion_point(module_scope)
