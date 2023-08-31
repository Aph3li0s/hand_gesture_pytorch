import torch
import model as m

class KeyPointClassifier(object):
    def __init__(self):
        self.device = 'cuda'
        self.model = m.SimpleNN2()
        self.model.load_state_dict(torch.load('models/model3_test.pth'))
        self.model.eval().to(self.device)

    def __call__(self, landmark_list):
        landmark_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(landmark_tensor)

        result_index = torch.argmax(output.squeeze()).item()
        return result_index
