from pathlib import Path


class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.resnet_model = None

    def load_yolo_model(self, model_path):
        """Load a YOLO model path if it exists."""
        self.yolo_model = self._load_existing_path(model_path)
        return self.yolo_model is not None

    def load_resnet_model(self, model_path):
        """Load a ResNet model path if it exists."""
        self.resnet_model = self._load_existing_path(model_path)
        return self.resnet_model is not None

    def _load_existing_path(self, model_path):
        path = Path(model_path)
        if path.exists():
            return str(path)
        return None
