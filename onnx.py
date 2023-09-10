import model as m
import torch

model = m.SimpleNN4()
model.load_state_dict(torch.load('models/9_9.pth'))
dummy_input = torch.zeros(1, 1, 126)

torch.onnx.export(model, dummy_input, '9_9_1.onnx', verbose=True)