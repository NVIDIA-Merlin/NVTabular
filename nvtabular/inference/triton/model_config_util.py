from google.protobuf.descriptor_pool import Default
from google.protobuf.message_factory import MessageFactory

def_pool = Default()
msg_factory = MessageFactory(pool=def_pool)
file_descriptor = def_pool.FindFileByName("model_config.proto")
if file_descriptor:
    ModelConfig = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelConfig"])
    ModelInput = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelInput"])
    ModelOutput = msg_factory.CreatePrototype(file_descriptor.message_types_by_name["ModelOutput"])
    TYPE_FP64 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_FP64"].number
    TYPE_FP32 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_FP32"].number
    TYPE_FP16 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_FP16"].number
    TYPE_INT64 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_INT64"].number
    TYPE_INT32 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_INT32"].number
    TYPE_INT16 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_INT16"].number
    TYPE_INT8 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_INT8"].number
    TYPE_UINT64 = (
        file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_UINT64"].number
    )
    TYPE_UINT32 = (
        file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_UINT32"].number
    )
    TYPE_UINT16 = (
        file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_UINT16"].number
    )
    TYPE_UINT8 = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_UINT8"].number
    TYPE_BOOL = file_descriptor.enum_types_by_name["DataType"].values_by_name["TYPE_BOOL"].number
