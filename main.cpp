#include <stdio.h>
#include <tchar.h>
#include <io.h>
#include "onnx2trt.h"
#include "trtInference.h"

//onnx�ļ���ַ����trt���治��������ȡonnxģ�Ͳ�����trt���棬�����trt������ֱ�Ӷ�ȡtrt����
string onnxFile1 = "./model/model.onnx";
//trt�����ļ���ַ�������治��������onnxģ�ʹ�����д��õ�ַ
string trtFile1 = "./model/model.trt";
string onnxFile2 = "./model/model2.onnx";
string trtFile2 = "./model/model2.trt";
//ͼƬ�ļ���
string path = "./data/";
string extention = ".jpg";
//string path = "E:/CODES/TensorFlow_OCR/dataset/chip_number2/train_images/";

extern int IMAGE_HEIGHT = 320;
extern int IMAGE_WIDTH = 320;
extern int IMAGE_CHANNEL = 3;
extern int BATCH_SIZE = 1;


void GetFilesO(string path, vector<string>& fileNames, string extention)
{
	//�ļ����
	intptr_t hFile = 0;
	//�ļ���Ϣ
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

	// ��ȡ�ļ�Ŀ¼
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
		readTrtFile(trtFile1, trtModelStream1);    //��trt�ļ��ж�ȡ���л����沢�����л���������л���trtModelStream��
		assert(trtModelStream1 != nullptr);
	}
	else
	{
		onnx2trt(onnxFile1, trtFile1, trtModelStream1);       // ��onnxģ��ת��Ϊtrtģ�Ͳ�����
		assert(trtModelStream1 != nullptr);
	}
	if (existEngine2) {

		readTrtFile(trtFile2, trtModelStream2);    //��trt�ļ��ж�ȡ���л����沢�����л���������л���trtModelStream��
		assert(trtModelStream2 != nullptr);
	}
	else
	{
		onnx2trt(onnxFile2, trtFile2, trtModelStream2);       // ��onnxģ��ת��Ϊtrtģ�Ͳ�����
		assert(trtModelStream2 != nullptr);
	}
	//do inference
	//doInferenceOnnx(trtModelStream, filenames);      //ִ��������ӡ���
	//doInferenceOnnx(trtModelStream, files);
	trtInference(trtModelStream1, trtModelStream2, filenames);

	system("pause");

	return 0;
}
