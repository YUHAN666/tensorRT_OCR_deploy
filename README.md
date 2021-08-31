# tensorRT_OCR_deploy

使用TensorRT部署TensorFlow模型，需要使用simple_save保存带参数的pb模型，再使用tf2onnx转化成onnx模型部署。

训练：TensorFlow1.13.1 cuda 10.0 cudnn7.6.5 部署：cuda10.2 tensorRT8.0.1.6 cudnn8.2

python -m tf2onnx.convert --saved-model E:/CODES/tensorflow_ocr/chip_pbmodel_1/ --output E:/CODES/tensorflow_ocr/chip_pbmodel_1/model.onnx --inputs image_input:0 --outputs dbnet/proba3_sigmoid:0

使用两个模型，第一个模型将字符块从背景中分割出来，第二个模型将字符块分割成一个个字符
