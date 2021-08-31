#pragma once
#include "NvInferRuntimeCommon.h"
#include <iostream>

using namespace std;
using namespace nvinfer1;
class Logger : public ILogger
{
public:

    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }

};

extern Logger gLogger;


