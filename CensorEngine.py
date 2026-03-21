import cv2
import numpy as np
from openvino import Core
import threading


class CensorEngine:
    all_possible_classes = ['ruler', 'barcode', 'colorcard', 'label', 'map', 'envelope', 'photo', 'attached_item', 'weights']

    def __init__(self, model_xml_path, censor_classes=None):
        """
        Args:
            model_xml_path: Path to the OpenVINO .xml model file.
            censor_classes: List of class name strings to censor. If None, all classes are censored.
        """
        if censor_classes is None:
            self.censor_classes = list(self.all_possible_classes)
        else:
            for c in censor_classes:
                if c not in self.all_possible_classes:
                    raise ValueError(f"Unknown class '{c}'. Valid classes: {self.all_possible_classes}")
            self.censor_classes = list(censor_classes)

        self.core = Core()
        self.model = self.core.read_model(model=model_xml_path)
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.inference_lock = threading.Lock()

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 640))
        img = resized.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def get_bounding_boxes(self, image):
        original_height, original_width = image.shape[:2]
        input_tensor = self.preprocess_image(image)
        with self.inference_lock:
            outputs = self.compiled_model([input_tensor])[self.output_layer]
        predictions = np.squeeze(outputs).T

        boxes, confidences, class_ids = [], [], []
        x_scale, y_scale = original_width / 640, original_height / 640

        for pred in predictions:
            box_coords, class_probs = pred[:4], pred[4:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]
            if confidence > 0.5:
                cx, cy, w, h = box_coords
                x1 = int((cx - w / 2) * x_scale); y1 = int((cy - h / 2) * y_scale)
                x2 = int((cx + w / 2) * x_scale); y2 = int((cy + h / 2) * y_scale)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.45)
        final_boxes = {name: [] for name in self.all_possible_classes}
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.all_possible_classes[class_ids[i]]
                final_boxes[class_name].append([x, y, x + w, y + h])
        return final_boxes

    def run(self, image_path, output_path=None):
        """
        Censor an image by blacking out detected bounding boxes for the specified classes.

        Args:
            image_path: Path to the input image.
            output_path: If provided, save the censored image to this path.

        Returns:
            The censored image as a numpy array (BGR).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        boxes_by_class = self.get_bounding_boxes(image)

        for class_name in self.censor_classes:
            for box in boxes_by_class.get(class_name, []):
                x1, y1, x2, y2 = box
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
                image[y1:y2, x1:x2] = 0

        if output_path:
            cv2.imwrite(output_path, image)

        return image


if __name__ == "__main__":
    model_path = "models/openvino/best.xml"
    image_to_process = "collage_test.jpg"

    # Censor only labels and barcodes
    engine = CensorEngine(model_path, censor_classes=["label", "barcode"])
    result = engine.run(image_to_process, output_path="censored_partial.jpg")
    print(f"Partial censor saved: {result.shape}")

    # Censor everything (default)
    engine_all = CensorEngine(model_path)
    result_all = engine_all.run(image_to_process, output_path="censored_all.jpg")
    print(f"Full censor saved: {result_all.shape}")