from google.protobuf.descriptor_pool import Default
from google.protobuf.message_factory import MessageFactory

def_pool = Default()
msg_factory = MessageFactory(pool=def_pool)
file_descriptor = def_pool.FindFileByName("model_config.proto")
if file_descriptor:
    ModelConfig = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelConfig"])
    ModelInput = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelInput"])
    ModelOutput = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelOutput"])
    ModelEnsembling = msg_factory.CreatePrototype(
        file_descriptor.message_types_by_name["ModelEnsembling"]
    )
    _DATATYPE = file_descriptor.enum_types_by_name["DataType"]
    TYPE_FP64 = _DATATYPE.values_by_name["TYPE_FP64"].number
    TYPE_FP32 = _DATATYPE.values_by_name["TYPE_FP32"].number
    TYPE_FP16 = _DATATYPE.values_by_name["TYPE_FP16"].number
    TYPE_INT64 = _DATATYPE.values_by_name["TYPE_INT64"].number
    TYPE_INT32 = _DATATYPE.values_by_name["TYPE_INT32"].number
    TYPE_INT16 = _DATATYPE.values_by_name["TYPE_INT16"].number
    TYPE_INT8 = _DATATYPE.values_by_name["TYPE_INT8"].number
    TYPE_UINT64 = _DATATYPE.values_by_name["TYPE_UINT64"].number
    TYPE_UINT32 = _DATATYPE.values_by_name["TYPE_UINT32"].number
    TYPE_UINT16 = _DATATYPE.values_by_name["TYPE_UINT16"].number
    TYPE_UINT8 = _DATATYPE.values_by_name["TYPE_UINT8"].number
    TYPE_BOOL = _DATATYPE.values_by_name["TYPE_BOOL"].number
    TYPE_STRING = _DATATYPE.values_by_name["TYPE_STRING"].number
