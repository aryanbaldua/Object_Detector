import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np

def main():
    # loading the pretrained models with default weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # evaluation mode, no training
    model.eval()
    
    # get caqtegory labels from weights metadata
    categories = weights.meta["categories"]
    print("Categories from TorchVision weights:", categories)

    # converts the frames to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # opening webcame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # convert BGR (OpenCV) to RGB (for PyTorch)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # transform for model inputs
        input_tensor = transform(rgb_frame)
        input_batch = [input_tensor]  # model expects a list of tensors

        # runs inference
        with torch.no_grad():
            outputs = model(input_batch)  # returns a list of dict

        # what is displayed on objects
        boxes = outputs[0]['boxes'].numpy()
        labels = outputs[0]['labels'].numpy()
        scores = outputs[0]['scores'].numpy()

        # to draw the boxes
        for box, label, score in zip(boxes, labels, scores):
            # only labels object if above 70% confidence
            if score < 0.7:
                continue

        
            class_name = categories[label]
            x1, y1, x2, y2 = box.astype(int)

            # draws the box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # label text
            text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # shows result to user
        cv2.imshow('FasterRCNN Detection', frame)

        # quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

