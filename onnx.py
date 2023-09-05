import model as m
import torch
model = m.SimpleNN4()
# Load pretrained weight

model.load_state_dict(torch.load('models/5_9_newest.pth'))

# Set dummy input
dummy_input = torch.zeros(1, 1, 126)

# Export to ONNX
torch.onnx.export(model, dummy_input, '5_9_afternoon.onnx', verbose=True)
