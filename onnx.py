import model as m
import torch
model = m.SimpleNN3()
# Load pretrained weight

model.load_state_dict(torch.load('models/6_9_4.pth'))

# Set dummy input
dummy_input = torch.zeros(1, 1, 126)

# Export to ONNX
torch.onnx.export(model, dummy_input, '6_9_4.onnx', verbose=True)
