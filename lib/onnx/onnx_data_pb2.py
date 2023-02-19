# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: onnx-data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from onnx import onnx_ml_pb2 as onnx_dot_onnx__ml__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='onnx-data.proto',
  package='onnx',
  syntax='proto2',
  serialized_options=b'H\003',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0fonnx-data.proto\x12\x04onnx\x1a\x12onnx/onnx-ml.proto\"\xf0\x02\n\rSequenceProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\telem_type\x18\x02 \x01(\x05\x12(\n\rtensor_values\x18\x03 \x03(\x0b\x32\x11.onnx.TensorProto\x12\x35\n\x14sparse_tensor_values\x18\x04 \x03(\x0b\x32\x17.onnx.SparseTensorProto\x12,\n\x0fsequence_values\x18\x05 \x03(\x0b\x32\x13.onnx.SequenceProto\x12\"\n\nmap_values\x18\x06 \x03(\x0b\x32\x0e.onnx.MapProto\x12,\n\x0foptional_values\x18\x07 \x03(\x0b\x32\x13.onnx.OptionalProto\"]\n\x08\x44\x61taType\x12\r\n\tUNDEFINED\x10\x00\x12\n\n\x06TENSOR\x10\x01\x12\x11\n\rSPARSE_TENSOR\x10\x02\x12\x0c\n\x08SEQUENCE\x10\x03\x12\x07\n\x03MAP\x10\x04\x12\x0c\n\x08OPTIONAL\x10\x05\"r\n\x08MapProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x10\n\x08key_type\x18\x02 \x01(\x05\x12\x0c\n\x04keys\x18\x03 \x03(\x03\x12\x13\n\x0bstring_keys\x18\x04 \x03(\x0c\x12#\n\x06values\x18\x05 \x01(\x0b\x32\x13.onnx.SequenceProto\"\xeb\x02\n\rOptionalProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\telem_type\x18\x02 \x01(\x05\x12\'\n\x0ctensor_value\x18\x03 \x01(\x0b\x32\x11.onnx.TensorProto\x12\x34\n\x13sparse_tensor_value\x18\x04 \x01(\x0b\x32\x17.onnx.SparseTensorProto\x12+\n\x0esequence_value\x18\x05 \x01(\x0b\x32\x13.onnx.SequenceProto\x12!\n\tmap_value\x18\x06 \x01(\x0b\x32\x0e.onnx.MapProto\x12+\n\x0eoptional_value\x18\x07 \x01(\x0b\x32\x13.onnx.OptionalProto\"]\n\x08\x44\x61taType\x12\r\n\tUNDEFINED\x10\x00\x12\n\n\x06TENSOR\x10\x01\x12\x11\n\rSPARSE_TENSOR\x10\x02\x12\x0c\n\x08SEQUENCE\x10\x03\x12\x07\n\x03MAP\x10\x04\x12\x0c\n\x08OPTIONAL\x10\x05\x42\x02H\x03'
  ,
  dependencies=[onnx_dot_onnx__ml__pb2.DESCRIPTOR,])



_SEQUENCEPROTO_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='onnx.SequenceProto.DataType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TENSOR', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SPARSE_TENSOR', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SEQUENCE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MAP', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OPTIONAL', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=321,
  serialized_end=414,
)
_sym_db.RegisterEnumDescriptor(_SEQUENCEPROTO_DATATYPE)

_OPTIONALPROTO_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='onnx.OptionalProto.DataType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TENSOR', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SPARSE_TENSOR', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SEQUENCE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MAP', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OPTIONAL', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=321,
  serialized_end=414,
)
_sym_db.RegisterEnumDescriptor(_OPTIONALPROTO_DATATYPE)


_SEQUENCEPROTO = _descriptor.Descriptor(
  name='SequenceProto',
  full_name='onnx.SequenceProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='onnx.SequenceProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='elem_type', full_name='onnx.SequenceProto.elem_type', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor_values', full_name='onnx.SequenceProto.tensor_values', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sparse_tensor_values', full_name='onnx.SequenceProto.sparse_tensor_values', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sequence_values', full_name='onnx.SequenceProto.sequence_values', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='map_values', full_name='onnx.SequenceProto.map_values', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optional_values', full_name='onnx.SequenceProto.optional_values', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SEQUENCEPROTO_DATATYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=414,
)


_MAPPROTO = _descriptor.Descriptor(
  name='MapProto',
  full_name='onnx.MapProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='onnx.MapProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='key_type', full_name='onnx.MapProto.key_type', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='keys', full_name='onnx.MapProto.keys', index=2,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='string_keys', full_name='onnx.MapProto.string_keys', index=3,
      number=4, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='values', full_name='onnx.MapProto.values', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=416,
  serialized_end=530,
)


_OPTIONALPROTO = _descriptor.Descriptor(
  name='OptionalProto',
  full_name='onnx.OptionalProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='onnx.OptionalProto.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='elem_type', full_name='onnx.OptionalProto.elem_type', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor_value', full_name='onnx.OptionalProto.tensor_value', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sparse_tensor_value', full_name='onnx.OptionalProto.sparse_tensor_value', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sequence_value', full_name='onnx.OptionalProto.sequence_value', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='map_value', full_name='onnx.OptionalProto.map_value', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optional_value', full_name='onnx.OptionalProto.optional_value', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _OPTIONALPROTO_DATATYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=533,
  serialized_end=896,
)

_SEQUENCEPROTO.fields_by_name['tensor_values'].message_type = onnx_dot_onnx__ml__pb2._TENSORPROTO
_SEQUENCEPROTO.fields_by_name['sparse_tensor_values'].message_type = onnx_dot_onnx__ml__pb2._SPARSETENSORPROTO
_SEQUENCEPROTO.fields_by_name['sequence_values'].message_type = _SEQUENCEPROTO
_SEQUENCEPROTO.fields_by_name['map_values'].message_type = _MAPPROTO
_SEQUENCEPROTO.fields_by_name['optional_values'].message_type = _OPTIONALPROTO
_SEQUENCEPROTO_DATATYPE.containing_type = _SEQUENCEPROTO
_MAPPROTO.fields_by_name['values'].message_type = _SEQUENCEPROTO
_OPTIONALPROTO.fields_by_name['tensor_value'].message_type = onnx_dot_onnx__ml__pb2._TENSORPROTO
_OPTIONALPROTO.fields_by_name['sparse_tensor_value'].message_type = onnx_dot_onnx__ml__pb2._SPARSETENSORPROTO
_OPTIONALPROTO.fields_by_name['sequence_value'].message_type = _SEQUENCEPROTO
_OPTIONALPROTO.fields_by_name['map_value'].message_type = _MAPPROTO
_OPTIONALPROTO.fields_by_name['optional_value'].message_type = _OPTIONALPROTO
_OPTIONALPROTO_DATATYPE.containing_type = _OPTIONALPROTO
DESCRIPTOR.message_types_by_name['SequenceProto'] = _SEQUENCEPROTO
DESCRIPTOR.message_types_by_name['MapProto'] = _MAPPROTO
DESCRIPTOR.message_types_by_name['OptionalProto'] = _OPTIONALPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SequenceProto = _reflection.GeneratedProtocolMessageType('SequenceProto', (_message.Message,), {
  'DESCRIPTOR' : _SEQUENCEPROTO,
  '__module__' : 'onnx_data_pb2'
  # @@protoc_insertion_point(class_scope:onnx.SequenceProto)
  })
_sym_db.RegisterMessage(SequenceProto)

MapProto = _reflection.GeneratedProtocolMessageType('MapProto', (_message.Message,), {
  'DESCRIPTOR' : _MAPPROTO,
  '__module__' : 'onnx_data_pb2'
  # @@protoc_insertion_point(class_scope:onnx.MapProto)
  })
_sym_db.RegisterMessage(MapProto)

OptionalProto = _reflection.GeneratedProtocolMessageType('OptionalProto', (_message.Message,), {
  'DESCRIPTOR' : _OPTIONALPROTO,
  '__module__' : 'onnx_data_pb2'
  # @@protoc_insertion_point(class_scope:onnx.OptionalProto)
  })
_sym_db.RegisterMessage(OptionalProto)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
