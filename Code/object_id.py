import cv2
import numpy as np

import cv2
import numpy as np

def obj_detection(object_img, scene_img, object_name):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_scene, des_scene = sift.detectAndCompute(scene_img, None)
    kp_object, des_object = sift.detectAndCompute(object_img, None)

    # Use a feature matcher (e.g., Brute-Force) to find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_object, des_scene, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches on the scene image
    img_matches = cv2.drawMatches(object_img, kp_object, scene_img, kp_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Get coordinates of matched points
    obj_pts = np.float32([kp_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scene_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, _ = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC)

    # Get object corners and transform them to scene coordinates
    h, w = object_img.shape[:2]
    obj_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    # Draw a bounding box around the detected object
    scene_img_with_box = scene_img.copy()
    cv2.polylines(scene_img_with_box, [np.int32(scene_corners)], True, (0, 255, 0), 2)

    # Annotate the detected object with the corresponding object name
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(scene_img_with_box, object_name, (int(scene_corners[0][0][0]), int(scene_corners[0][0][1]) - 10),
                font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    return scene_img_with_box

def table_data(scenes, total_objects):
    # Calculate metrics for each scene
    scene_metrics = []
    for scene in scenes:
        metrics = calculate_metrics(scene["actual_objects"], scene["identified_objects"], total_objects)
        scene_metrics.append({"scene_id": scene["scene_id"], **metrics})

    # Calculate average metrics for the complete dataset
    avg_metrics = {metric: sum(scene[metric] for scene in scene_metrics) / len(scene_metrics) for metric in scene_metrics[0]}

    # Display results
    print("\nMetrics for Each Scene:")
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Scene", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1-Score"
    ))
    for scene in scene_metrics:
        print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            scene["scene_id"],
            scene["TP"],
            scene["FP"],
            scene["TN"],
            scene["FN"],
            scene["Precision"],
            scene["Recall"],
            scene["F1-Score"]
        ))

    print("\nAverage Metrics for the Complete Dataset:")
    print("{:<15} {:<15} {:<15} {:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
        "TP", "FP", "TN", "FN", avg_metrics["Precision"], avg_metrics["Recall"], avg_metrics["F1-Score"], avg_metrics["Accuracy"]
    ))

def calculate_metrics(actual_objects, identified_objects, total_objects):
    true_positives = len(set(actual_objects) & set(identified_objects))
    false_positives = len(set(identified_objects) - set(actual_objects))
    true_negatives = total_objects - len(set(actual_objects) | set(identified_objects))
    false_negatives = len(set(actual_objects) - set(identified_objects))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

    return {
        "TP": true_positives,
        "FP": false_positives,
        "TN": true_negatives,
        "FN": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Accuracy": accuracy
    }