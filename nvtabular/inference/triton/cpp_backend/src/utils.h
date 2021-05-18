// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef UTILS_H
#define UTILS_H

#include "triton/backend/backend_common.h"

namespace triton {
namespace backend {
namespace nvtabular {

class Utils {
public:
  static std::string ConvertToNumpyType(TRITONSERVER_DataType dtype) {
    if (dtype == TRITONSERVER_TYPE_INVALID)
      return std::string("invalid");
    else if (dtype == TRITONSERVER_TYPE_BOOL)
      return std::string("np.bool_");
    else if (dtype == TRITONSERVER_TYPE_UINT8)
      return std::string("np.uint8");
    else if (dtype == TRITONSERVER_TYPE_UINT16)
      return std::string("np.uint16");
    else if (dtype == TRITONSERVER_TYPE_UINT32)
      return std::string("np.uint32");
    else if (dtype == TRITONSERVER_TYPE_UINT64)
      return std::string("np.uint64");
    else if (dtype == TRITONSERVER_TYPE_INT8)
      return std::string("np.int8");
    else if (dtype == TRITONSERVER_TYPE_INT16)
      return std::string("np.int16");
    else if (dtype == TRITONSERVER_TYPE_INT32)
      return std::string("np.int32");
    else if (dtype == TRITONSERVER_TYPE_INT64)
      return std::string("np.int64");
    else if (dtype == TRITONSERVER_TYPE_FP16)
      return std::string("np.float16");
    else if (dtype == TRITONSERVER_TYPE_FP32)
      return std::string("np.float32");
    else if (dtype == TRITONSERVER_TYPE_FP64)
      return std::string("np.float64");
    else
      return std::string("np.bytes");
  }

  static TRITONSERVER_DataType ConvertToTritonType(std::string &output_dtype) {
    if (output_dtype.compare("TYPE_INVALID") == 0)
      return TRITONSERVER_TYPE_INVALID;
    else if (output_dtype.compare("TYPE_BOOL") == 0)
      return TRITONSERVER_TYPE_BOOL;
    else if (output_dtype.compare("TYPE_UINT8") == 0)
      return TRITONSERVER_TYPE_UINT8;
    else if (output_dtype.compare("TYPE_UINT16") == 0)
      return TRITONSERVER_TYPE_UINT16;
    else if (output_dtype.compare("TYPE_UINT32") == 0)
      return TRITONSERVER_TYPE_UINT32;
    else if (output_dtype.compare("TYPE_UINT64") == 0)
      return TRITONSERVER_TYPE_UINT64;
    else if (output_dtype.compare("TYPE_INT8") == 0)
      return TRITONSERVER_TYPE_INT8;
    else if (output_dtype.compare("TYPE_INT16") == 0)
      return TRITONSERVER_TYPE_INT16;
    else if (output_dtype.compare("TYPE_INT32") == 0)
      return TRITONSERVER_TYPE_INT32;
    else if (output_dtype.compare("TYPE_INT64") == 0)
      return TRITONSERVER_TYPE_INT64;
    else if (output_dtype.compare("TYPE_FP16") == 0)
      return TRITONSERVER_TYPE_FP16;
    else if (output_dtype.compare("TYPE_FP32") == 0)
      return TRITONSERVER_TYPE_FP32;
    else if (output_dtype.compare("TYPE_FP64") == 0)
      return TRITONSERVER_TYPE_FP64;
    else
      return TRITONSERVER_TYPE_BYTES;
  }

  static size_t GetTritonTypeByteSize(TRITONSERVER_DataType dtype) {
    if (dtype == TRITONSERVER_TYPE_INVALID)
      return 0;
    else if (dtype == TRITONSERVER_TYPE_BOOL)
      return sizeof(bool);
    else if (dtype == TRITONSERVER_TYPE_UINT8)
      return sizeof(uint8_t);
    else if (dtype == TRITONSERVER_TYPE_UINT16)
      return sizeof(uint16_t);
    else if (dtype == TRITONSERVER_TYPE_UINT32)
      return sizeof(uint32_t);
    else if (dtype == TRITONSERVER_TYPE_UINT64)
      return sizeof(uint64_t);
    else if (dtype == TRITONSERVER_TYPE_INT8)
      return sizeof(int8_t);
    else if (dtype == TRITONSERVER_TYPE_INT16)
      return sizeof(int16_t);
    else if (dtype == TRITONSERVER_TYPE_INT32)
      return sizeof(int32_t);
    else if (dtype == TRITONSERVER_TYPE_INT64)
      return sizeof(int64_t);
    else if (dtype == TRITONSERVER_TYPE_FP16)
      return 2;
    else if (dtype == TRITONSERVER_TYPE_FP32)
      return sizeof(float);
    else if (dtype == TRITONSERVER_TYPE_FP64)
      return sizeof(double);
    else
      return 1;
  }

  static size_t GetMaxStringLen(const unsigned char *str, const uint64_t len) {
    size_t max_len = 0;
    size_t pad_size = 4;
    uint64_t i = 0;

    while (i < len) {
      size_t curr = (size_t)str[i];
      if (curr > max_len) {
        max_len = curr;
      }

      i += pad_size + curr;
      if (i > len) {
        max_len = -1;
      }
    }
    return max_len;
  }

  static void ConstructNumpyStringArray(wchar_t *dest, const uint64_t elem_len,
                                        const unsigned char *source,
                                        const uint64_t len) {
    size_t pad_size = 4;
    uint64_t i = 0;
    uint64_t j = 0;

    while (i < len) {
      size_t curr = (size_t)source[i];
      for (size_t k = 0; k < curr; ++k) {
        dest[j + k] = (wchar_t)source[i + pad_size + k];
      }
      i += pad_size + curr;
      j += elem_len;
    }
  }
};

} // namespace nvtabular
} // namespace backend
} // namespace triton

#endif /* UTILS_H */
