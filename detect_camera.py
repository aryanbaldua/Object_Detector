import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np

def main():
    # 1. Load the pretrained model with default weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # Switch to evaluation mode (no training, no targets required)
    model.eval()
    
    # 2. Get the official category labels directly from the weights metadata
    categories = weights.meta["categories"]
    print("Categories from TorchVision weights:", categories)

    # 3. Define a transformation to convert frames to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 4. Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Convert BGR (OpenCV) to RGB (for PyTorch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform for model input
        input_tensor = transform(rgb_frame)
        input_batch = [input_tensor]  # model expects a list of tensors

        # 5. Run inference
        with torch.no_grad():
            outputs = model(input_batch)  # returns a list of dict

        # The output dictionary keys: ['boxes', 'labels', 'scores']
        boxes = outputs[0]['boxes'].numpy()
        labels = outputs[0]['labels'].numpy()
        scores = outputs[0]['scores'].numpy()

        # 6. Draw bounding boxes for confident detections
        for box, label, score in zip(boxes, labels, scores):
            # Filter out low-confidence
            if score < 0.7:
                continue

            # Use the official categories list from the weights
            class_name = categories[label]

            # box is [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.astype(int)

            # Draw a green bounding box for all objects
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 7. Show the result
        cv2.imshow('FasterRCNN Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

