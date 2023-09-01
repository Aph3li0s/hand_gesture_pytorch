import model as m
import torch
model = m.SimpleNN2()
# Load pretrained weight

model.load_state_dict(torch.load('models/model3_test.pth'))

# Set dummy input
dummy_input = torch.zeros(1, 1, 63)

# Export to ONNX
torch.onnx.export(model, dummy_input, 'onnx_model.onnx', verbose=True)

