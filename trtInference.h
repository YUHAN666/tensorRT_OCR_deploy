#pragma once

#include "NvInfer.h"
#include "logger.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"
#include <stdlib.h>
#include <memory>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>


using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

extern int IMAGE_HEIGHT;
extern int IMAGE_WIDTH;
extern int IMAGE_CHANNEL;
extern int BATCH_SIZE;

#define CHECK(status)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                    \
        if (ret != 0)                                                                                               \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                   \
            abort();                                                                                                 \
        }                                                                                                              \
    } while (0)


inline int64_t volume(const nvinfer1::Dims& d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF: return 2;
	case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}


Mat prepareImage(Mat& img);
Mat prepareImage2(Mat& img);


void doInferenceOnnx(IHostMemory* trtModelStream, vector<string> files);
void trtInference(IHostMemory* trtModelStream1, IHostMemory* trtModelStream2, vector<string> files);

bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream);  //output buffer for the TensorRT model

void viewEngineBindings(ICudaEngine* engine);

Mat getMaskCoordinate(Mat& mask, Mat& image);
Mat getMaskCoordinate2(Mat& mask, Mat& img);	//Ë«Ä£ÐÍ³¢ÊÔ