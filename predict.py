import torch
import model as m
import torch.nn.functional as F

class KeyPointClassifier(object):
    def __init__(self):
        self.device = 'cuda'
        self.model = m.SimpleNN2()
        self.model.load_state_dict(torch.load('models/test.pth'))
        self.model.eval().to(self.device)

    def __call__(self, landmark_list):
        landmark_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(landmark_tensor)

        probabilities = F.softmax(output, dim=1)
        max_prob, max_class = torch.max(probabilities, dim=1)
        threshold = 0.1  # Adjust this threshold as needed
        # print(max_class)
        # for i, prob in enumerate(probabilities[0]):
        #     if abs(prob - max_prob) < threshold:
        #         print(f"Class {i}: Probability = {prob.item()}")
        result_index = max_class.item()
        return result_index