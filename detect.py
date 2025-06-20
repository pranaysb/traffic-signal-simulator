import cv2
import numpy as np
import time
import os

# Let us load our yolov3 model. now
net = cv2.dnn.readNet("yolo-tiny-cfg/yolov3-tiny.weights", "yolo-tiny-cfg/yolov3-tiny.cfg")

# code to load the class labels or like names which has like cars trucks etc 
with open("yolo-tiny-cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# getting the output layers from the model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

lane_images = {
    "North": "lane1.jpg",
    "South": "lane2.jpg",
    "East": "lane3.jpg",
    "West": "lane4.jpg"
}

# Create output directory if needed
os.makedirs("output", exist_ok=True)

# Function to count vehicles in an image using NMS
def count_vehicles(image_path, show_boxes=False, save_output=True, lane_name=""):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return 0

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    count = len(indices)

    if show_boxes and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        window_name = f"{lane_name} Lane - Detected Vehicles"
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    if save_output and len(indices) > 0:
        output_path = f"output/{lane_name.lower()}_output.jpg"
        cv2.imwrite(output_path, image)

    return count

# Count vehicles in each lane
lane_vehicle_counts = {}
for lane, path in lane_images.items():
    count = count_vehicles(path, show_boxes=True, save_output=True, lane_name=lane)
    lane_vehicle_counts[lane] = count
    print(f"{lane} Lane → 🚗 Vehicles: {count}")

# Total green signal time to distribute
total_time = 120
total_vehicles = sum(lane_vehicle_counts.values())

# Allocate green signal time per lane
lane_timings = {}
if total_vehicles == 0:
    print("⚠️ No vehicles detected in any lane. All signals set to 0 seconds.")
    for lane in lane_vehicle_counts:
        lane_timings[lane] = 0
else:
    for lane, count in lane_vehicle_counts.items():
        allocated_time = int((count / total_vehicles) * total_time)
        lane_timings[lane] = allocated_time
        print(f"⏱️ {lane} Lane → Green Signal Time: {allocated_time} seconds")

# colors for each direction
lane_colors = {
    "North": (0, 255, 0),
    "South": (0, 0, 255),
    "East": (255, 0, 0),
    "West": (0, 255, 255)
}

# simulation part
print("\n🚦 Starting Smart Traffic Signal Simulation...\n")
for lane, duration in lane_timings.items():
    image = cv2.imread(lane_images[lane])
    if image is None:
        print(f"❌ Image not found: {lane_images[lane]}")
        continue

    height, width = image.shape[:2]
    vis_image = image.copy()

    # Display simulation text
    cv2.putText(vis_image, f"{lane} LANE - GREEN for {duration}s",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_colors[lane], 2)
    cv2.rectangle(vis_image, (10, 100), (width - 10, height - 10), lane_colors[lane], 10)

    cv2.imshow("Smart Traffic Signal Simulation", vis_image)
    print(f"🟢 {lane} lane has GREEN signal for {duration} seconds.")
    for i in range(duration, 0, -1):
        print(f"   ⏳ {lane} lane: {i} sec remaining...", end="\r")
        time.sleep(1)

    print(f"🔴 {lane} lane signal ended.\n")
    cv2.waitKey(1000)

cv2.destroyAllWindows()