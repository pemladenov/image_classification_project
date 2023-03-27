import onnx

model_path = 'squeezenet1.0-12-int8.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
