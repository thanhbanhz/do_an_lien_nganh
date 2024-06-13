import cv2
import random
import matplotlib.pyplot as plt
def visualize_results_usual_yolo_inference(
        img,
        result,
        show_boxes=True,
        show_class=True,
        color_class_background=(0, 0, 255),
        color_class_text=(255, 255, 255),
        thickness=4,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.5,
        delta_colors=0,
        dpi=150,
        random_object_colors=False,
        show_confidences=False,
        axis_off=True,
        show_classes_list=[],
        return_image_array=True
):
    """
    Visualizes the results of usual YOLOv8 or YOLOv8-seg inference on an image

    Args:
        img (numpy.ndarray): The input image in BGR format.
        model: The object detection or segmentation model (yolov8).
        imgsz (int): The input image size for the model. Default is 640.
        conf (float): The confidence threshold for detection. Default is 0.5.
        iou (float): The intersection over union threshold for detection. Default is 0.7.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple): The background bgr color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher).
        random_object_colors (bool): If True, colors for each object are selected randomly.
        show_confidences (bool): If True and show_class=True, confidences near class are visualized.
        axis_off (bool): If True, axis is turned off in the final visualization.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
        return_image_array (bool): If True, the function returns the image bgr array instead of displaying it.
                                   Default is False.

    Returns:
        None/np.array
    """

    labeled_image = img.copy()

    if random_object_colors:
        random.seed(int(delta_colors))

    # Process each prediction
    for pred in result:

        class_names = pred.names

        # Get the bounding boxes and convert them to a list of lists
        boxes = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        classes = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        confidences = pred.boxes.conf.cpu().numpy()

        num_objects = len(classes)

        # Visualization
        for i in range(num_objects):
            # Get the class for the current detection
            class_index = int(classes[i])
            class_name = class_names[class_index]

            if show_classes_list and class_index not in show_classes_list:
                continue

            if random_object_colors:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                # Assign color according to class
                random.seed(int(classes[i] + delta_colors))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            box = boxes[i]
            x_min, y_min, x_max, y_max = box


            # Write class label
            if show_boxes:
                cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

            if show_class:
                if show_confidences:
                    label = f'{str(class_name)} {confidences[i]:.2}'
                else:
                    label = str(class_name)
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(
                    labeled_image,
                    (x_min, y_min),
                    (x_min + text_width + 5, y_min + text_height + 5),
                    color_class_background,
                    -1,
                )
                cv2.putText(
                    labeled_image,
                    label,
                    (x_min + 5, y_min + text_height),
                    font,
                    font_scale,
                    color_class_text,
                    thickness=thickness,
                )

    if return_image_array:
        return labeled_image
    else:
        # Display the final image with overlaid masks and labels
        plt.figure(figsize=(8, 8), dpi=dpi)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.imshow(labeled_image)
        if axis_off:
            plt.axis('off')
        plt.show()
