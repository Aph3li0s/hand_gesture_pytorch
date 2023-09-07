import model as m
import torch
model = m.SimpleNN4()
# Load pretrained weight

model.load_state_dict(torch.load('models/7_9_3.pth'))

# Set dummy input
dummy_input = torch.zeros(1, 1, 126)

# Export to ONNX
torch.onnx.export(model, dummy_input, '7_9_4.onnx', verbose=True)
