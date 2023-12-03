import cv2
import numpy as np

import cv2
import numpy as np

def scene_objects(scene_img, object_images, object_info):
    obj_keys = []
    matches = []
    scene_dupe = scene_img.copy()

    obj_index = 0
    for obj in object_images:
        obj_name = object_info[obj_index]['name']
        detect_img, scene_key, obj_key, match = obj_detection(obj, scene_img, obj_name, scene_dupe)

        if obj_key is None:
            obj_keys.append(None)
        else:
            obj_keys.append({
                'img': obj_key,
                'number': obj_index + 1
            })

        if match is None:
            matches.append(None)
        else:
            matches.append({
                'img': match,
                'number': obj_index + 1
            })

        obj_index += 1
    return detect_img, scene_key, obj_keys, matches

def obj_detection(object_img, scene_img, object_name, scene_untouched):
    # Initialize AKAZE detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_scene, des_scene = sift.detectAndCompute(scene_untouched, None)
    kp_object, des_object = sift.detectAndCompute(object_img, None)

    scene_dupe = scene_untouched.copy()

    scene_keypoints = cv2.drawKeypoints(scene_dupe, kp_scene, None)
    #object_keypoints = cv2.drawKeypoints(object_img, kp_object, None)
    object_keypoints = None

    # Use a feature matcher (e.g., Brute-Force) to find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_object, des_scene, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get coordinates of matched points
    obj_pts = np.float32([kp_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scene_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if not obj_pts.any() or not scene_pts.any() or len(good_matches) < 4:
        print("No matches found / Not enough matches found")
        return scene_img, scene_keypoints, object_keypoints, None

    img_matches = cv2.drawMatches(object_img, kp_object, scene_img, kp_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Use RANSAC to estimate the homography matrix
    H, _ = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC)
    if H is None:
        # Try create a homography matrix with all points
        H, _ = cv2.findHomography(obj_pts, scene_pts, 0)
        if H is None:
            print("Failed to create homography matrix")
            return scene_img, scene_keypoints, object_keypoints, img_matches

    # Choose the first set of matched keypoints
    obj_corners = np.float32([[0, 0], [0, object_img.shape[0] - 1],
                              [object_img.shape[1] - 1, object_img.shape[0] - 1], [object_img.shape[1] - 1, 0]])

    obj_corners = obj_corners.reshape(4, 1, 2)
    # Transform object corners to the scene using the estimated homography
    scene_corners = cv2.perspectiveTransform(obj_corners.reshape(-1, 1, 2), H)

    # Calculate the center of the detected object
    center_x, center_y = np.mean(scene_corners.squeeze(), axis=0)

    # Draw a fixed-size square (50x50) around the detected object at the center
    square_size = 50
    x, y = int(center_x), int(center_y)
    cv2.rectangle(scene_img, (x - square_size // 2, y - square_size // 2),
                  (x + square_size // 2, y + square_size // 2), (0, 255, 0), 2)

    # Annotate the detected object with the corresponding object name at the center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # Adjust the font scale as needed
    text_size = cv2.getTextSize(object_name, font, font_scale, 2)[0]
    text_x, text_y = int(center_x - text_size[0] // 2), int(center_y + text_size[1] // 2)
    cv2.putText(scene_img, object_name, (text_x, text_y),
                font, font_scale, (255, 0, 0), 2, cv2.LINE_AA)

    return scene_img, scene_keypoints, object_keypoints, img_matches

def update_scene_data(scene_img, scene_info):
    # We are given a scene image with detected objects and the scene info
    # First, show the user the scene image with the detected objects
    print("Press any key to continue...")
    cv2.imshow("Scene Image", scene_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    identified_objects = input("Enter the correctly identified objects: ")
    identified_objects = [int(obj) for obj in identified_objects.split(",")]

    misidentified_objects = input("Enter the incorrectly identified objects: ")
    misidentified_objects = [int(obj) for obj in misidentified_objects.split(",")]

    # Update the scene info with the identified and misidentified objects
    scene_info["identified_objects"] = identified_objects
    scene_info["misidentified_objects"] = misidentified_objects

    return scene_info

def table_data(scenes, total_objects):
    # Calculate metrics for each scene
    scene_metrics = []
    for scene in scenes:
        metrics = calculate_metrics(scene["actual_objects"], scene["identified_objects"], scene["misidentified_objects"], total_objects)
        scene_metrics.append({"scene_id": scene["scene_id"], **metrics})

    # Calculate average metrics for the complete dataset
    avg_metrics = {
        metric: sum(scene[metric] for scene in scene_metrics) / len(scene_metrics) 
        for metric in scene_metrics[0]
        if metric != "scene_id"
    }

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

def calculate_metrics(actual_objects, identified_objects, misidentified_objects, total_objects):
    true_positives = len(set(actual_objects) & set(identified_objects))
    false_positives = len(set(identified_objects) - set(actual_objects))
    true_negatives = total_objects - len(set(actual_objects) | set(identified_objects) | set(misidentified_objects))
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