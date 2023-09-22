import onnx
import onnxruntime
import cv2
import numpy as np
from PIL import Image, ImageDraw
path = "testing\YoloV7_Tiny.onnx"

# Create an ONNX Runtime inference session
ort_session = onnxruntime.InferenceSession(path)
image = Image.open("testing/Africa-2.jpg")
im = image.resize((320, 320))  # Resize to the model's input size
print(*im)
# image = np.array(im, dtype=np.float32) / 255.0  # Normalize the image
# image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
# image = image[np.newaxis, :] 
# input_name = ort_session.get_inputs()[0].name
# input_tensor = {input_name: image}

# # Run inference
# outputs = ort_session.run(None, input_tensor)
# output = np.array(outputs, dtype=np.float32)
# print(output)
# boxes = output[:,:,:-1].flatten().astype(int) 
# confidences = output[:, :,-1] 
# print(boxes)
# print(len(outputs))
