#include <stdio.h>
#include <tchar.h>
#include <io.h>
#include "onnx2trt.h"
#include "trtInference.h"

//onnx文件地址，如trt引擎不存在则会读取onnx模型并建立trt引擎，如存在trt引擎则直接读取trt引擎
string onnxFile1 = "./model/model.onnx";
//trt引擎文件地址，如引擎不存在则会从onnx模型创建并写入该地址
string trtFile1 = "./model/model.trt";
string onnxFile2 = "./model/model2.onnx";
string trtFile2 = "./model/model2.trt";
//图片文件夹
string path = "./data/";
string extention = ".jpg";
//string path = "E:/CODES/TensorFlow_OCR/dataset/chip_number2/train_images/";

extern int IMAGE_HEIGHT = 320;
extern int IMAGE_WIDTH = 320;
extern int IMAGE_CHANNEL = 3;
extern int BATCH_SIZE = 1;


void GetFilesO(string path, vector<string>& fileNames, string extention)
{
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("/*" + extention).c_str(), &fileinfo)) != -1)
	{
		do
		{
			string q;
			q.append(path).append("/").append(fileinfo.name);
			fileNames.push_back(q);
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}


extern "C" __declspec(dllexport) int main()
{

	vector<string> filenames;

	// 读取文件目录
	GetFilesO(path, filenames, extention);
	//wstring n = L"n";
	//for (int i = 0; i < files.size(); i++)
	//{
	//	wcout << (files[i].substr(106,1)==n) << endl;
	//}
	//system("pause");


	 //create a TensorRT model from the onnx model and serialize it to a stream
	IHostMemory* trtModelStream1{ nullptr };
	IHostMemory* trtModelStream2{ nullptr };

	// create and load engine
	fstream existEngine1, existEngine2;
	existEngine1.open(trtFile1, ios::in);
	existEngine2.open(trtFile2, ios::in);
	if (existEngine1)
	{
		readTrtFile(trtFile1, trtModelStream1);    //从trt文件中读取序列化引擎并反序列化，最后序列化到trtModelStream中
		assert(trtModelStream1 != nullptr);
	}
	else
	{
		onnx2trt(onnxFile1, trtFile1, trtModelStream1);       // 将onnx模型转化为trt模型并保存
		assert(trtModelStream1 != nullptr);
	}
	if (existEngine2) {

		readTrtFile(trtFile2, trtModelStream2);    //从trt文件中读取序列化引擎并反序列化，最后序列化到trtModelStream中
		assert(trtModelStream2 != nullptr);
	}
	else
	{
		onnx2trt(onnxFile2, trtFile2, trtModelStream2);       // 将onnx模型转化为trt模型并保存
		assert(trtModelStream2 != nullptr);
	}
	//do inference
	//doInferenceOnnx(trtModelStream, filenames);      //执行推理并打印输出
	//doInferenceOnnx(trtModelStream, files);
	trtInference(trtModelStream1, trtModelStream2, filenames);

	system("pause");

	return 0;
}
