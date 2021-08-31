#include "trtInference.h"


bool readTrtFile(const std::string& engineFile, //name of the engine file
	IHostMemory*& trtModelStream)  //output buffer for the TensorRT model 
//��trtģ�Ͷ�ȡ���ڴ���
{
	using namespace std;
	fstream file;
	cout << "loading filename from:" << engineFile << endl;
	nvinfer1::IRuntime* trtRuntime;
	file.open(engineFile, ios::binary | ios::in);
	file.seekg(0, ios::end);
	int length = file.tellg();
	//cout << "length:" << length << endl;
	file.seekg(0, ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	cout << "load engine done" << endl;
	std::cout << "deserializing" << endl;
	trtRuntime = createInferRuntime(gLogger);
	ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length);
	assert(engine != nullptr);
	cout << "deserialize done" << endl;
	trtModelStream = engine->serialize();

	return true;
}


void doInferenceOnnx(IHostMemory* trtModelStream, vector<string> files)
{
	// get engine
	try
	{
		assert(trtModelStream != nullptr);
		IRuntime* runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);

		// ������������
		ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
		assert(engine != nullptr);
		viewEngineBindings(engine);
		std::cout << "Successfully deserialized Engine" << std::endl;

		trtModelStream->destroy();
		IExecutionContext* context = engine->createExecutionContext();
		assert(context != nullptr);

		// ��ȡ�������ݵ����������������
		assert(engine->getNbBindings() == 2);   // ��ģ��ֻ��һ�������һ�����
		void* gpuBuffers[2];       // GPU�������ڴ�ָ�� *buffer[0]: input; *buffer[1]: output
		std::vector<int64_t> bufferSize;        // �������ڴ��С ��ΪNbBindings�飬��Ӧÿһ����������
		int nbBindings = engine->getNbBindings();
		bufferSize.resize(nbBindings);

		for (int i = 0; i < nbBindings; ++i)
		{
			nvinfer1::Dims dims = engine->getBindingDimensions(i);      // Binding��ά��
			nvinfer1::DataType dtype = engine->getBindingDataType(i);   // Binding����������
			int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);       // ����ά�Ⱥ��������ͼ����Binding������ڴ��С
			bufferSize[i] = totalSize;
			CHECK(cudaMalloc(&gpuBuffers[i], totalSize));      // ��GPU��ΪBinding������Ӧ��С���ڴ�
		}

		// ����CUDA����ִ�д��ƶ�
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));       // ����CUDA��, CUDA������һ������Ĺ������У�GPU �Ӹö���������ȡ��������ִ��
		//int falseAccount = 0;

		for (int i = 0; i < files.size() - 1; i++)
		{
			// ��ȡͼƬ
			auto t_start_pre = std::chrono::high_resolution_clock::now();

			cv::Mat img = cv::imread(files[i], 1);
			Mat img_float = prepareImage(img);
			float* d = (float*)img_float.data;
			auto t_end_pre = std::chrono::high_resolution_clock::now();
			float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
			std::cout << "prepare image take: " << total_pre << " ms." << endl;

			// ������ͼƬ��curInput�첽���Ƶ�GPU�ڴ���,cudaMemcpyAsync����ʱ����֤�����Ѿ���ϣ������Ҫʹ��enqueueV2�����ʹ��execute����ʹ��ͬ������cudaMemcpy
			//CHECK(cudaMemcpy(gpuBuffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice));
			auto t_start_1 = std::chrono::high_resolution_clock::now();
			CHECK(cudaMemcpyAsync(gpuBuffers[0], d, bufferSize[0], cudaMemcpyHostToDevice, stream));
			auto t_end_1 = std::chrono::high_resolution_clock::now();
			float total_1 = std::chrono::duration<float, std::milli>(t_end_1 - t_start_1).count();
			std::cout << "copy image to gpu takes: " << total_1 << " ms." << endl;
			// ִ������
			auto t_start = std::chrono::high_resolution_clock::now();
			//context->execute(1, gpuBuffers);
			//context->executeV2(gpuBuffers);
			context->enqueueV2(gpuBuffers, stream, nullptr);
			//cudaStreamSynchronize(stream);      //�ȴ���ִ�����
			auto t_end = std::chrono::high_resolution_clock::now();
			float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
			std::cout << "Inference take: " << total << " ms." << endl;

			// �������������out

			int outSize = bufferSize[1] / sizeof(float);
			float* out = new float[outSize];
			//��ģ�����������gpu buffer�п�����������out�У���ģ��ֻ��һ��output node
			auto t_start_2 = std::chrono::high_resolution_clock::now();
			CHECK(cudaMemcpyAsync(out, gpuBuffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
			auto t_end_2 = std::chrono::high_resolution_clock::now();
			float total_2 = std::chrono::duration<float, std::milli>(t_end_2 - t_start_2).count();
			std::cout << "copy image from gpu takes: " << total_2 << " ms." << endl;
			cudaStreamSynchronize(stream);      //�ȴ���ִ�����

			auto t_start_cv = std::chrono::high_resolution_clock::now();
			// ��֤���mask���
			cv::Mat mask_out = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, out);
			auto t_end_cv = std::chrono::high_resolution_clock::now();
			float total_cv = std::chrono::duration<float, std::milli>(t_end_cv - t_start_cv).count();
			std::cout << "opencv get image take: " << total_cv << " ms." << endl;
			float maxValue = *max_element(mask_out.begin<float>(), mask_out.end<float>());
			cv::Mat m = getMaskCoordinate(mask_out, img);
			std::cout << "\n" << std::endl;
			//for (int batch = 0; batch < BATCH_SIZE; batch++) {

			//	std::cout << "Output: " << out[batch] << endl;	

			//}
			//��֤���
			//string p = "p";
			//string n = "n";
			//if ((files[i].substr(106, 1) == p) && (out[0] < 0.5)) {
			//	falseAccount += 1;
			//}
			//else if ((files[i].substr(106, 1) == n) && (out[0] > 0.5)) {
			//	falseAccount += 1;
			//}
			//cout << "\n" << endl;

			//inputImgs.clear();
		}


		//cout << "False Account: " << falseAccount << " out of " << files.size() << " images" << endl;
		//cout << "Accuracy: " << 1 - float(falseAccount) / float(files.size()) << endl;

		// release the stream and the buffers

		cudaStreamDestroy(stream);
		CHECK(cudaFree(gpuBuffers[0]));
		CHECK(cudaFree(gpuBuffers[1]));

		// destroy the engine
		context->destroy();
		engine->destroy();
		runtime->destroy();
	}
	catch (const std::exception&)
	{
		
	}

}


void trtInference(IHostMemory* trtModelStream1, IHostMemory* trtModelStream2, vector<string> files) {


	assert(trtModelStream1 != nullptr);
	assert(trtModelStream2 != nullptr);
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);

	// ������������
	ICudaEngine* engine1 = runtime->deserializeCudaEngine(trtModelStream1->data(), trtModelStream1->size());
	assert(engine1 != nullptr);
	viewEngineBindings(engine1);
	std::cout << "Successfully deserialized Engine1" << std::endl;
	ICudaEngine* engine2 = runtime->deserializeCudaEngine(trtModelStream2->data(), trtModelStream2->size());
	assert(engine2 != nullptr);
	viewEngineBindings(engine2);
	std::cout << "Successfully deserialized Engine2" << std::endl;

	trtModelStream1->destroy();
	trtModelStream2->destroy();
	IExecutionContext* context1 = engine1->createExecutionContext();
	assert(context1 != nullptr);
	IExecutionContext* context2 = engine2->createExecutionContext();
	assert(context2 != nullptr);

	// ��ȡ�������ݵ����������������
	assert(engine1->getNbBindings() == 2);   // ��ģ��ֻ��һ�������һ�����
	assert(engine2->getNbBindings() == 2);   // ��ģ��ֻ��һ�������һ�����
	void* gpuBuffers1[2];       // GPU�������ڴ�ָ�� *buffer[0]: input; *buffer[1]: output
	void* gpuBuffers2[2];       // GPU�������ڴ�ָ�� *buffer[0]: input; *buffer[1]: output
	std::vector<int64_t> bufferSize1;        // �������ڴ��С ��ΪNbBindings�飬��Ӧÿһ����������
	std::vector<int64_t> bufferSize2;        // �������ڴ��С ��ΪNbBindings�飬��Ӧÿһ����������
	int nbBindings1 = engine1->getNbBindings();
	int nbBindings2 = engine2->getNbBindings();
	bufferSize1.resize(nbBindings1);
	bufferSize2.resize(nbBindings2);

	for (int i = 0; i < nbBindings1; ++i)
	{
		nvinfer1::Dims dims1 = engine1->getBindingDimensions(i);      // Binding��ά��
		nvinfer1::DataType dtype1 = engine1->getBindingDataType(i);   // Binding����������
		int64_t totalSize1 = volume(dims1) * 1 * getElementSize(dtype1);       // ����ά�Ⱥ��������ͼ����Binding������ڴ��С
		bufferSize1[i] = totalSize1;
		CHECK(cudaMalloc(&gpuBuffers1[i], totalSize1));      // ��GPU��ΪBinding������Ӧ��С���ڴ�
	}
	// ����CUDA����ִ�д��ƶ�
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));       // ����CUDA��, CUDA������һ������Ĺ������У�GPU �Ӹö���������ȡ��������ִ��



	for (int i = 0; i < nbBindings2; ++i)
	{
		nvinfer1::Dims dims2 = engine2->getBindingDimensions(i);      // Binding��ά��
		nvinfer1::DataType dtype2 = engine2->getBindingDataType(i);   // Binding����������
		int64_t totalSize2 = volume(dims2) * 1 * getElementSize(dtype2);       // ����ά�Ⱥ��������ͼ����Binding������ڴ��С
		bufferSize2[i] = totalSize2;
		CHECK(cudaMalloc(&gpuBuffers2[i], totalSize2));      // ��GPU��ΪBinding������Ӧ��С���ڴ�
	}



	for (int i = 0; i < files.size() - 1; i++)
	{
		// ��ȡͼƬ

		cv::Mat img = cv::imread(files[i], 1);
		Mat img_float = prepareImage(img);
		float* d = (float*)img_float.data;
		// ������ͼƬ��curInput�첽���Ƶ�GPU�ڴ���,cudaMemcpyAsync����ʱ����֤�����Ѿ���ϣ������Ҫʹ��enqueueV2�����ʹ��execute����ʹ��ͬ������cudaMemcpy
		//CHECK(cudaMemcpy(gpuBuffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice));
		CHECK(cudaMemcpyAsync(gpuBuffers1[0], d, bufferSize1[0], cudaMemcpyHostToDevice, stream));
		context1->enqueueV2(gpuBuffers1, stream, nullptr);
		// �������������out
		int outSize = bufferSize1[1] / sizeof(float);
		float* out = new float[outSize];
		CHECK(cudaMemcpyAsync(out, gpuBuffers1[1], bufferSize1[1], cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);      //�ȴ���ִ�����
		// ��֤���mask���
		cv::Mat mask_out = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, out);
		cv::Mat img_2 = getMaskCoordinate(mask_out, img);

		std::cout << "\n" << std::endl;

		Mat img_float2 = prepareImage2(img_2);
		float* d2 = (float*)img_float2.data;

		CHECK(cudaMemcpyAsync(gpuBuffers2[0], d2, bufferSize2[0], cudaMemcpyHostToDevice, stream));
		context2->enqueueV2(gpuBuffers2, stream, nullptr);
		// �������������out
		int outSize2 = bufferSize2[1] / sizeof(float);
		float* out2 = new float[outSize2];
		CHECK(cudaMemcpyAsync(out2, gpuBuffers2[1], bufferSize2[1], cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);      //�ȴ���ִ�����
		// ��֤���mask���
		cv::Mat mask_out2 = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, out2);
		cv::Mat number_img = getMaskCoordinate2(mask_out2, img_2);

	}

}

Mat prepareImage2(Mat& img) {

	cv::Mat img_float, resized_img, pad_img;
	cv::Size size = cv::Size(320, 160);
	cv::resize(img, resized_img, size);
	cv::copyMakeBorder(resized_img, pad_img, 0, 160, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
	pad_img.convertTo(img_float, CV_32FC3, 1 / 255.0, 0.0);		//��һ��

	return img_float;
}


Mat prepareImage(Mat& img) {

	cv::Mat img_float, resized_img, pad_img;
	cv::Size size = cv::Size(320, 240);
	cv::resize(img, resized_img, size);
	cv::copyMakeBorder(resized_img, pad_img, 0, 80, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
	pad_img.convertTo(img_float, CV_32FC3, 1 / 255.0, 0.0);		//��һ��

	return img_float;
}



void viewEngineBindings(ICudaEngine* engine)
{
	int nbBindings = engine->getNbBindings();
	std::cout << "Number Bindings: " << nbBindings << std::endl;
	for (int i = 0; i < nbBindings; ++i) {

		string bindingName = engine->getBindingName(i);
		Dims bindingDim = engine->getBindingDimensions(i);
		string bindingFormat = engine->getBindingFormatDesc(i);
		if (engine->bindingIsInput(i)) {
			std::cout << "Input Binding: " << bindingName << std::endl;
		}
		else {
			std::cout << "Output Binding: " << bindingName << std::endl;
		}
		std::cout << "Binding Dim: (" << std::to_string(bindingDim.d[0]) + "," + std::to_string(bindingDim.d[1]) + "," + std::to_string(bindingDim.d[2]) + "," + std::to_string(bindingDim.d[3]) + ")" << std::endl;
		std::cout << "Binding Format: " << bindingFormat << std::endl;
	}
}


Mat getMaskCoordinate(Mat& mask, Mat& img) {

	std::vector< std::vector< cv::Point> > contours;
	mask = mask * 255;
	float scale = 6.4;
	Mat binary_mask, threshold_mask;
	threshold(mask, threshold_mask, 125, 255, cv::THRESH_BINARY);
	threshold_mask.convertTo(binary_mask, CV_8UC1);

	findContours(binary_mask, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	int xmax = 0;
	int xmin = 320;
	int ymax = 0;
	int ymin = 320;
	for (int i = 0; i < contours[0].size(); i++) {

		if (contours[0][i].x > xmax) {
			xmax = contours[0][i].x;
		}
		if (contours[0][i].x < xmin)
		{
			xmin = contours[0][i].x;
		}

		if (contours[0][i].y > ymax) {
			ymax = contours[0][i].y;
		}
		if (contours[0][i].y < ymin)
		{
			ymin = contours[0][i].y;
		}

	}
	int xcenter = int((xmin+xmax)/2*scale);
	int ycenter = int(((ymin + ymax) / 2)*scale);
	cv::drawContours(img, contours, -1, cv::Scalar::all(255));
	vector<Range> range;
	range.push_back(Range(ycenter -80, ycenter +80));
	range.push_back(Range(xcenter -160, xcenter +160));
	Mat result_image = img(range);
	//cv::imshow("result_image", result_image);
	//cv::waitKey();


	return result_image;
}


Mat getMaskCoordinate2(Mat& mask, Mat& img) {

	std::vector< std::vector< cv::Point> > contours;
	mask = mask * 255;
	float scale = 1;
	Mat binary_mask, threshold_mask;
	threshold(mask, threshold_mask, 125, 255, cv::THRESH_BINARY);
	threshold_mask.convertTo(binary_mask, CV_8UC1);

	findContours(binary_mask, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	
	int xmax = 0;
	int xmin = 320;
	int ymax = 0;
	int ymin = 320;
	for (int j = 0; j < contours.size(); j++) {

		cv::drawContours(img, contours, j, (0, 0, 255));
	}

	//for (int i = 0; i < contours[0].size(); i++) {

	//	if (contours[0][i].x > xmax) {
	//		xmax = contours[0][i].x;
	//	}
	//	if (contours[0][i].x < xmin)
	//	{
	//		xmin = contours[0][i].x;
	//	}

	//	if (contours[0][i].y > ymax) {
	//		ymax = contours[0][i].y;
	//	}
	//	if (contours[0][i].y < ymin)
	//	{
	//		ymin = contours[0][i].y;
	//	}

	//}
	//int xcenter = int((xmin + xmax) / 2 * scale);
	//int ycenter = int(((ymin + ymax) / 2) * scale);
	//cv::drawContours(img, contours, -1, cv::Scalar::all(255));
	//vector<Range> range;
	//range.push_back(Range(ycenter - 80, ycenter + 80));
	//range.push_back(Range(xcenter - 160, xcenter + 160));
	//Mat result_image = img(range);
	cv::imshow("img", img);
	cv::waitKey();


	return img;
}