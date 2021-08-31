#pragma once

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"
#include "logger.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

using namespace nvinfer1;
using namespace std;


bool onnx2trt(string onnxFile, string trtFile, IHostMemory*& trtModelStream);
