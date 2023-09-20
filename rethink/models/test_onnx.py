import torch
import onnx
from onnx2torch import convert
from rethink.preprocess import Loader, Trimmer, Padder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to ONNX model
onnx_model_path = "/media/magalhaes/sound/onnx/yolov7_tiny_distilled_export.onnx"
onnx_model = onnx.load(onnx_model_path)
model = convert(onnx_model)
model.to(device)
model.eval()


loader = Loader(22050, device="cuda:0" if torch.cuda.is_available() else "cpu")
trimmer = Trimmer(10, 22050)
padder = Padder(10, 22050)

filename = "../test/2023-03-20 15_40_17.502605+00_00.wav"

x = loader(filename)
x = trimmer(x)
x = padder(x)
# add batch dimension
x = x.unsqueeze(0)

y = model(x)
print(y)
