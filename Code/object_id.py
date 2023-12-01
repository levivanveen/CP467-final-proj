import cv2

def obj_detection(objects, scenes):
    print("Made it here")
    # Load the scene and object images
    scene_img = scenes[0]
    object_img = objects[0]

    print(scene_img is None)
    print(object_img is None)

    # debug by showing the images
    cv2.imshow('Scene', scene_img)
    cv2.waitKey(0)
    cv2.imshow('Object', object_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    # Display the result
    print("helllo")
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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