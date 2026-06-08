import os


class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.resnet_model = None

    def load_yolo_model(self, model_path):
        """Load a YOLO model path if it exists."""
        if os.path.exists(model_path):
            self.yolo_model = model_path
            return True
        return False

    def load_resnet_model(self, model_path):
        """Load a ResNet model path if it exists."""
        if os.path.exists(model_path):
            self.resnet_model = model_path
            return True
        return False
