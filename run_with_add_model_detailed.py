from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
import numpy as np
from typing import Tuple

video_writer = None

home_team = "Twente"
away_team = "Fenerbahçe"
referee_label = "Referee"
home_hex = "#cc242e"
away_hex = "#ebf1ff"

ROBOFLOW_API_KEY = ""

home_lower = np.array([0, 100, 100])
home_upper = np.array([10, 255, 255])

away_lower = np.array([0, 0, 200])
away_upper = np.array([180, 25, 255])

referee_lower = np.array([40, 100, 100])
referee_upper = np.array([80, 255, 255])

video_path = r"video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def filter_team_by_color(image, bbox, lower_color, upper_color, threshold=80):
    x1, y1, x2, y2 = bbox
    player_region = image[y1:y2, x1:x2]

    player_region_blurred = cv2.GaussianBlur(player_region, (5, 5), 0)

    hsv_image = cv2.cvtColor(player_region_blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    white_pixel_count = np.sum(mask == 255)
    
    if white_pixel_count > threshold:
        return True

    return False

def hex_to_hsv_range(hex_color: str, tolerance: int = 10):
    hex_color = hex_color.lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    bgr_color_np = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_color_np, cv2.COLOR_BGR2HSV)[0][0]

    lower_bound = np.array([max(0, hsv_color[0] - tolerance), 50, 50])
    upper_bound = np.array([min(180, hsv_color[0] + tolerance), 255, 255])

    return lower_bound, upper_bound

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    global video_writer

    detections = sv.Detections.from_inference(predictions)
    image = video_frame.image.copy()

    labels = []
    
    home_rgb = hex_to_rgb(home_hex)
    home_color = sv.Color.from_rgb_tuple(home_rgb)
    away_rgb = hex_to_rgb(away_hex)
    away_color = sv.Color.from_rgb_tuple(away_rgb)

    for i, detection in enumerate(detections.xyxy):
        class_id = detections.class_id[i]
        
        bbox = detection.astype(int)

        if class_id == 0:
            labels.append("")
            continue
        
        if class_id == 1:
            labels.append(referee_label)
            continue

        if class_id == 2:
            if (filter_team_by_color(image, bbox, home_lower, home_upper)):
                labels.append(home_team)
            elif (filter_team_by_color(image, bbox, away_lower, away_upper)):
                labels.append(away_team)
            else:
                labels.append("")
                continue
        
        else:
            labels.append("")
            continue

    ball_detections = detections[detections.class_id == 0]
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.YELLOW, position=sv.Position.BOTTOM_CENTER)
    image = triangle_annotator.annotate(image, detections=ball_detections)
    
    player_detections = detections[detections.class_id == 2]
    xyxy_values = player_detections.xyxy
    
    referee_detections = detections[detections.class_id == 3]
    referee_xyxy_values = referee_detections.xyxy

    for i, xyxy in enumerate(xyxy_values):
        bbox = xyxy.astype(int).reshape(1, -1)

        class_ids = np.array([1])
        detections = sv.Detections(bbox, class_id=class_ids)

        if filter_team_by_color(image, bbox[0], home_lower, home_upper):
            ellipse_annotator = sv.EllipseAnnotator(color=home_color)
            label_annotator = sv.RichLabelAnnotator(text_color=home_color, text_padding=10, text_position=sv.Position.TOP_CENTER, font_path="fonts/Poppins-Bold.ttf")
            label = home_team
        elif filter_team_by_color(image, bbox[0], away_lower, away_upper):
            ellipse_annotator = sv.EllipseAnnotator(color=away_color)
            label_annotator = sv.RichLabelAnnotator(text_color=away_color, text_padding=10, text_position=sv.Position.TOP_CENTER, font_path="fonts/Poppins-Bold.ttf")
            label = away_team
        else:
            continue         

        image = ellipse_annotator.annotate(image.copy(), detections=detections)
        image = label_annotator.annotate(image.copy(), detections=detections, labels=[label])
        
    for i, xyxy in enumerate(referee_xyxy_values):
        bbox = xyxy.astype(int).reshape(1, -1)

        class_ids = np.array([1])
        detections = sv.Detections(bbox, class_id=class_ids)

        if filter_team_by_color(image, bbox[0], referee_lower, referee_upper):
            ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.GREEN)
            label_annotator = sv.RichLabelAnnotator(text_color=sv.Color.GREEN, text_padding=10, text_position=sv.Position.TOP_CENTER, font_path="fonts/Poppins-Bold.ttf")
            label = referee_label
        else:
            continue         

        image = ellipse_annotator.annotate(image.copy(), detections=detections)
        image = label_annotator.annotate(image.copy(), detections=detections, labels=[label])
        
        
    if video_writer is None:
        video_writer = cv2.VideoWriter(
            "output_video.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    video_writer.write(image)

    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    #model_id="football-player-detection-kucab/6",
    model_id="football-players-detection-3zvbc/12",
    video_reference="video.mp4",
    on_prediction=my_custom_sink,
    api_key=ROBOFLOW_API_KEY
)

pipeline.start()
pipeline.join()

if video_writer is not None:
    video_writer.release()
    cap.release()

cv2.destroyAllWindows()
