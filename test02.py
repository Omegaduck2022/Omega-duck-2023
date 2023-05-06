import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Curl counter variables
counter = 0
stage = None


def calculate_angle(a, b, c):

    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


confThreshold = 0.6
classesFile = 'coco80.names'
classes = []
# Load all classes in coco80.names into classes[]
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    print(classes)
    print(len(classes))
net = cv2.dnn.readNetFromDarknet('yolov3-320.cfg','yolov3-320.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



# VIDEO FEED
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW);
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        height, width, ch = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        print(layerNames)
        output_layers_names = net.getUnconnectedOutLayersNames()
        print(output_layers_names)
        LayerOutputs = net.forward(output_layers_names)
        print(len(LayerOutputs))
        bboxes = []  # array for all bounding boxes of detected classes
        confidences = []  # array for all confidence values of matching detected classes
        class_ids = []  # array for all class IDs of matching detected classes

        for output in LayerOutputs:
            for detection in output:
                scores = detection[5:]  # omit the first 5 values
                class_id = 39
                confidence = scores[class_id]
                if confidence > confThreshold:
                    center_x = int(detection[0] * width)  # YOLO predicts centers of image
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    bboxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)


        indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)  # Non-maximum suppresion
        # print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(bboxes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = bboxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                errorPan = (x + w / 2) - 640 / 2

            # print(type(pos))
            time.sleep(0.1)

        cv2.imshow('Mediapipe Feed', frame)



        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                print(landmarks)
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

        except:
                pass

        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()






