
import math
import cv2
import numpy as np

def process_objects(info, color_image, depth_image, classnames,JOINT_NAMES,CONFID_THRESHOLD,FONT,
                    FONT_SCALE,TEXT_COLOR, FONT_THICKNESS,model_type):
    local_objects = []
    # print("yes#######################################data",info.keypoints.data)
    # print("yes#######################################xy",info.keypoints.xy)
    # print("yes#######################################xyn",info.keypoints.xyn)
    if model_type !=str('obb'):
        for box in info.boxes:
            confidence = int(box.conf[0] * 100)
            if confidence > CONFID_THRESHOLD:
                class_detect = classnames[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.putText(color_image, f'{class_detect,confidence}', (x1 + 8, y1 - 12), FONT, FONT_SCALE,
                                (TEXT_COLOR), FONT_THICKNESS)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(depth_image, f'{class_detect,confidence}', (x1 + 8, y1 - 12), FONT, FONT_SCALE,
                                (TEXT_COLOR), FONT_THICKNESS)
                cv2.rectangle(depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # local_objects.append((x1, y1, x2 - x1, y2 - y1, center_x, center_y, confidence, class_detect))
                if model_type == str('pose_estimation'):
                    data_np = info.keypoints.cpu().numpy()
                    xy_np = info.keypoints.xy.cpu().numpy()
                    xyn_np = info.keypoints.xyn.cpu().numpy()
                    confidence_np = info.keypoints.conf.cpu().numpy()
                    for point in xy_np:
                        for i,p in enumerate(point):
                            x, y = int(p[0]), int(p[1])
                            cv2.circle(color_image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
                            cv2.putText(color_image, f'{JOINT_NAMES[i]}', (x + 8, y - 12), FONT, FONT_SCALE,
                                (TEXT_COLOR), FONT_THICKNESS)
                            # cv2.circle(depth_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                
                if model_type == str('segmentation'):
                    #print("ClassName#################################", class_detect)
                    #print("class#########################################",info.masks.x
                    masks = info.masks.xy  # Assuming masks are available in info object

                    for mask in masks:
                        # Convert mask to a list of numpy arrays with integer dtype
                        mask_np = np.array(mask, dtype=np.int32)

                        # Reshape mask_np to have a shape of (n, 1, 2)
                        mask_np = mask_np.reshape((-1, 1, 2))

                        # Draw filled polygon on the original image
                        cv2.fillPoly(depth_image, [mask_np], color=(0, 0, 255))
    # else:
    #     # boxes = info.boxes.xyxy  # Bounding boxes in xyxy format
    #     # confs = info.boxes.conf  # Confidence scores
    #     classes = info.boxes.cls  # Class labels

    #     print(classes)
    return local_objects, color_image, depth_image


def process_objects1(info, color_image,classnames,JOINT_NAMES,CONFID_THRESHOLD,FONT,
                    FONT_SCALE,TEXT_COLOR, FONT_THICKNESS,model_type):
    local_objects = []
    # print("yes#######################################data",info.keypoints.data)
    # print("yes#######################################xy",info.keypoints.xy)
    # print("yes#######################################xyn",info.keypoints.xyn)
    if model_type !=str('obb'):
        for box in info.boxes:
            confidence = int(box.conf[0] * 100)
            if confidence > CONFID_THRESHOLD:
                class_detect = classnames[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.putText(color_image, f'{class_detect,confidence}', (x1 + 8, y1 - 12), FONT, FONT_SCALE,
                                (TEXT_COLOR), FONT_THICKNESS)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                local_objects.append((x1, y1, x2 - x1, y2 - y1, center_x, center_y, confidence, class_detect))
                if model_type == str('pose_estimation'):
                    data_np = info.keypoints.cpu().numpy()
                    xy_np = info.keypoints.xy.cpu().numpy()
                    xyn_np = info.keypoints.xyn.cpu().numpy()
                    confidence_np = info.keypoints.conf.cpu().numpy()
                    for point, confidences in zip(xy_np, confidence_np):
                        for i, (p, conf) in enumerate(zip(point, confidences)):
                            x, y = int(p[0]), int(p[1])
                            if conf > (75/100) and JOINT_NAMES[i]== "lf_eye"or JOINT_NAMES[i]== "rt_eye"or JOINT_NAMES[i]=="lt_wrist"or JOINT_NAMES[i]=="rt_wrist":
                                cv2.circle(color_image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
                                cv2.putText(color_image, f'{JOINT_NAMES[i]}: {conf:.2f}', (x + 8, y - 12), FONT, FONT_SCALE,
                                            (TEXT_COLOR), FONT_THICKNESS)
                    
                if model_type == str('segmentation'):
                    #print("ClassName#################################", class_detect)
                    #print("class#########################################",info.masks.x
                    masks = info.masks.xy  # Assuming masks are available in info object

                    for mask in masks:
                        # Convert mask to a list of numpy arrays with integer dtype
                        mask_np = np.array(mask, dtype=np.int32)

                        # Reshape mask_np to have a shape of (n, 1, 2)
                        mask_np = mask_np.reshape((-1, 1, 2))

                        # Draw filled polygon on the original image
                        cv2.fillPoly(color_image, [mask_np], color=(0, 0, 255))
    # else:
    #     # boxes = info.boxes.xyxy  # Bounding boxes in xyxy format
    #     # confs = info.boxes.conf  # Confidence scores
    #     classes = info.boxes.cls  # Class labels

    #     print(classes)
    return local_objects, color_image