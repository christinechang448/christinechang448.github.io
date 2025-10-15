import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def select_video_file():
    videos = ["Video 1: Night", "Video 2: Curved Roads", "Video 3: Straight Roads"]
    print(f"Available videos: {videos}")

    user_vid_choice = input("Please pick a video from the available options (PLS type full name): ")
    if user_vid_choice in videos:
        print(f"You have chosen {user_vid_choice}")
    else:
        print(f"{user_vid_choice} does not exist in the possible list of videos")

    if user_vid_choice == "Video 1: Night":
        video_path = "videos/vid1_night.mov"
    elif user_vid_choice == "Video 2: Curved Roads":
        video_path = "videos/vid2_curved.mov"
    else:
        video_path = "videos/vid3_day_straight.mov"
    return video_path

def load_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    
    video_info = {
        "capture": cap,
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_path": video_path
    }
    
    print(f"Video Loaded: {video_path}")
    print(f"Resolution: {video_info['frame_width']}x{video_info['frame_height']}")
    print(f"FPS: {video_info['fps']}, Total Frames: {video_info['total_frames']}")

    return video_info

def region_of_interest_mask(image, frame_width, frame_height):
    height, width = image.shape[:2]
    
    vertices = np.array([
        (50, frame_height + 500),                   
        (1100, frame_height + 500), 
        (550, 250)
    ], dtype = np.int32)
    
   
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_edges(image, frame_width, frame_height): 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) #watch video on kernels and arrays later
    edges = cv2.Canny(blur, 50, 150) #watch video on how canny works later
    roi_edges = region_of_interest_mask(edges, frame_width, frame_height)
    
    return roi_edges

def detect_lines_hough(edges):
    lines = cv2.HoughLinesP(
        edges,
        rho = 1,              #Finer distance resolution
        theta = np.pi/180,    #Angular resolution in radians
        threshold = 50,       #Lower threshold means detect more lines
        minLineLength = 30,   #Shorter minimum line length
        maxLineGap = 60       #Allow larger gaps between segments
    )
    
    if lines is not None:
        return lines
    else:
        return []

def separate_left_right_lines(lines):
    left_lines = []
    right_lines = []
    
    if len(lines) == 0:  
        return left_lines, right_lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 - x1 == 0:  
            continue
        slope = (y2 - y1) / (x2 - x1)
        
       
        if slope < -0.3:  #Left lane (negative slope) it weird. I watch video
            left_lines.append(line[0])
        elif slope > 0.3:  #Right lane (positive slope)
            right_lines.append(line[0])
        
    
    return left_lines, right_lines

def average_line_segments(lines, frame_height):
    if not lines:
        return None
    
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        x_coords.extend([x1, x2]) 
        y_coords.extend([y1, y2])
    
    if len(x_coords) < 2:
        return None
    
    poly = np.polyfit(y_coords, x_coords, 1) #polyfit finds the best line that flows through all the x and y coordinates

    y1 = frame_height
    y2 = int(frame_height * 0.6) #0.6 of the way up just feels right
    x1 = int(poly[0] * y1 + poly[1]) #poly[0] is the slope and poly [1] is y-int. so basically we do math x = my + int
    x2 = int(poly[0] * y2 + poly[1])
    
    return x1, y1, x2, y2

def calculate_steering_angle(left_line, right_line, frame_width, frame_height):
    if left_line is None and right_line is None:
        return 0.0 
    frame_center = frame_width // 2
    
    if left_line is not None and right_line is not None:
        left_x1, _, left_x2, _ = left_line
        right_x1, _, right_x2, _ = right_line
        lane_center = (left_x1 + right_x1) // 2
    elif left_line is not None:
        left_x1, _, _, _ = left_line
        lane_center = left_x1 + 200  
    else:
        right_x1, _, _, _ = right_line
        lane_center = right_x1 - 200  #change the standard lane width later
    
    angle = np.arctan2(lane_center - frame_center, frame_height * 0.5) #i will draw picture if you want. need diagram to explain
    angle_degrees = np.degrees(angle) #np.arctan2 returns radians not degrees. brain no picture radians
    
    return angle_degrees

def draw_lane_lines(frame, left_line, right_line):
    overlay = frame.copy()
    
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)  #Green for left lane
    
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 5)  #Red for right lane
    
    return overlay

def add_information_overlay(frame, steering_angle, frame_num, total_frames):
    overlay = frame.copy()
    
    text = f"Steering Angle: {steering_angle:.2f} degrees"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame_text = f"Frame: {frame_num}/{total_frames}"
    cv2.putText(overlay, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if steering_angle > 3:
        direction = "Right"
    elif steering_angle < -3:
        direction = "Left"
    else: 
        direction = "Straight"

    cv2.putText(overlay, f"Direction: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay

def process_single_frame(frame, frame_width, frame_height):
    edges = detect_edges(frame, frame_width, frame_height)
    lines = detect_lines_hough(edges)
    left_lines, right_lines = separate_left_right_lines(lines)
    left_line = average_line_segments(left_lines, frame_height)
    right_line = average_line_segments(right_lines, frame_height)
    steering_angle = calculate_steering_angle(left_line, right_line, frame_width, frame_height)
    
    return left_line, right_line, steering_angle

def process_entire_video(video_info, output_path):
    cap = video_info['capture']  
    frame_width = video_info['frame_width']
    frame_height = video_info['frame_height']
    fps = video_info['fps']
    total_frames = video_info['total_frames']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    steering_angles = []
    frame_numbers = []
    timestamps = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_line, right_line, steering_angle = process_single_frame(frame, frame_width, frame_height)

        steering_angles.append(steering_angle)
        frame_numbers.append(frame_count)
        timestamps.append(frame_count / fps)
 
        frame_with_lanes = draw_lane_lines(frame, left_line, right_line)
        final_frame = add_information_overlay(frame_with_lanes, steering_angle, frame_count, total_frames)
        
        out.write(final_frame)
        
        frame_count += 1
        
    cap.release()
    out.release()
    
    return output_path, steering_angles, frame_numbers, timestamps

def create_steering_angle_graph(steering_angles, frame_numbers):
    pio.renderers.default = "browser"
    
    df = pd.DataFrame({
        'Frame': frame_numbers,
        'Steering Angle (degrees)': steering_angles
    })

    df['Steering Change (deg/frame)'] = df['Steering Angle (degrees)'].diff() #take the derivative to get the rate at which the steering angle changes

    fig = make_subplots(
        rows = 2, 
        cols = 1,
        subplot_titles =(
            'Steering Angle vs Frame Number', 
            'Steering Angle Change Per Frame'
        ),
        vertical_spacing = 0.5
    )

    fig.add_trace(
        go.Scatter(
            x=df['Frame'],
            y=df['Steering Angle (degrees)'],
            mode = 'lines',
            name = 'Steering Angle',
            line = dict(color = 'blue', width = 2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['Frame'][1:],  #Skip first frame, no change in the data
            y=df['Steering Change (deg/frame)'][1:],
            mode='lines',
            name = 'Angle Change',
            line= dict(color='red', width=2)
        ),
        row=2, col=1
    )

    fig.update_layout(
        title_text = "Steering Angle Analysis",
        height = 600,
    )
    
    fig.update_xaxes(title_text = "Frame Number", row=1, col=1)
    fig.update_yaxes(title_text = "Steering Angle (degrees)", row=1, col=1)
    
    fig.update_xaxes(title_text = "Frame Number", row=2, col=1)
    fig.update_yaxes(title_text = "Angle Change (deg/frame)", row=2, col=1)
  
    fig.show()
    
def main():
    print("Starting Lane Detection System")
    
    video_path = select_video_file()
    output_path = 'output_with_lanes.mp4'
    
    print(f"Input video: {video_path}")
    
    video_info = load_video_info(video_path)
    output_video, steering_angles, frame_numbers, timestamps = process_entire_video(video_info, output_path)
    
    create_steering_angle_graph(steering_angles, frame_numbers)
    print(f"Output video: {output_video}")
        
    video_info['capture'].release()  
    cv2.destroyAllWindows()


main()