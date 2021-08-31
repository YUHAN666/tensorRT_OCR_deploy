#include "onnx2trt.h"

bool onnx2trt(string onnxFile, string trtFile, IHostMemory*& trtModelStream) {

	IBuilder* builder = createInferBuilder(gLogger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

	std::cout << "Successfully read Onnx model" << std::endl;

	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}

	IBuilderConfig* config = builder->createBuilderConfig();
	IOptimizationProfile* profile = builder->createOptimizationProfile();
	//profile->setDimensions
	config->setMaxWorkspaceSize(400000);
	//config->addOptimizationProfile()
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	std::cout << "Successfully build Engine" << std::endl;

	parser->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
	//IHostMemory* serializedModel = engine->serialize();
	trtModelStream = engine->serialize();
	std::ofstream file;
	file.open(trtFile, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)trtModelStream->data(), trtModelStream->size());
	cout << "save engine file done to " << trtFile << endl;


	file.close();
	//serializedModel->destroy();

	return true;

}
