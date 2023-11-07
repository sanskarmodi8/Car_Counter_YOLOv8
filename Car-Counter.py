from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Open the video file for processing
cap = cv2.VideoCapture("cars.mp4")

# Load the YOLO model for object detection
model = YOLO("yolov8l.pt")

# Define class names for object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load a mask image that defines the unmasked area
mask = cv2.imread("mask.png")

# Initialize the SORT tracker with specific parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# tracker: An instance of the SORT (Simple Online and Realtime Tracking) tracker.
# max_age: Maximum number of frames to keep an object alive without new detections.
# min_hits: Minimum number of detections required to start tracking an object.
# iou_threshold: Intersection over Union (IOU) threshold for associating new detections with existing objects.

# Define the region limits where vehicle counts are recorded
limits = [400, 297, 673, 297]

# Create an empty list to store vehicle IDs for counting
totalCount = []

while True:
    # img reads the next frame from the  while success stores the boolean whether success in reading or not
    success, img = cap.read()

    # Apply the mask to create a region of interest
    imgRegion = cv2.bitwise_and(img, mask)

    # Display the count of cars
    cvzone.putTextRect(img, f'Count : {len(totalCount)}', (0,35),
                       scale=3, thickness=3, offset=10, colorR=(255,0,255))

    # Use the YOLO model for object detection in the region of interest
    results = model(imgRegion, stream=True)

    # Create an empty array to store object detections
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Extract confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the object is a car with sufficient confidence
            if currentClass =='car' and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Use the SORT tracker to update object tracking
    resultsTracker = tracker.update(detections)

    # Draw a line to mark the region limits
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process tracked objects
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding boxes around tracked objects
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        # Display the object ID on the frame
        cvzone.putTextRect(img, f'id : #{int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=8)

        # center of the bbox
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if an object crosses the specified limits
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15: #we used +-15 to set range in when the center of the bbox of the car comes, it gets counted
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) #change the color of line to green when count


    # Display the processed frame
    cv2.imshow("Image", img)

    # Wait for a key press (or a specific duration) to continue processing the next frame
    # cv2.waitKey(0)
    cv2.waitKey(1) # wait for 1 millisecond
