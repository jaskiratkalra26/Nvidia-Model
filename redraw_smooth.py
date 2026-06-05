import cv2
import os
import re
import glob

def parse_bbox(text, width, height):
    results = []
    parts = text.split('<ref>')
    for part in parts[1:]:
        if '</ref>' not in part:
            continue
        label, rest = part.split('</ref>', 1)
        
        matches = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', rest)
        for m in matches:
            x1, y1, x2, y2 = map(int, m)
            # Filter out giant boxes that cover the whole hand/screen (>35% of frame)
            # The coordinates are on a 1000x1000 scale, so max area is 1,000,000.
            if (x2 - x1) * (y2 - y1) > 350000:
                continue
            if x1 <= 1000 and y1 <= 1000 and x2 <= 1000 and y2 <= 1000:
                px1 = int((x1 / 1000.0) * width)
                py1 = int((y1 / 1000.0) * height)
                px2 = int((x2 / 1000.0) * width)
                py2 = int((y2 / 1000.0) * height)
                results.append({"label": label.strip(), "box": (px1, py1, px2, py2)})
    return results

def redraw_smooth():
    input_video = "surgery_video.mp4"
    log_file = "processing.log"
    output_dir = "output_frames_smooth"
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found!")
        return
        
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Parsing log file...")
    
    alpha = 0.3 # Smoothing factor (0.0 to 1.0). Lower = smoother but lags behind fast movement.
    
    frame_to_raw_boxes = {}
    last_raw_boxes = {}
    current_frame = 0
    
    # 1. Parse all the raw boxes from the log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "Processing frame" in line:
                m = re.search(r'Processing frame (\d+)', line)
                if m:
                    current_frame = int(m.group(1))
                    frame_to_raw_boxes[current_frame] = {}
                    last_raw_boxes = {}
            elif line.startswith("Output for "):
                m = re.match(r"Output for '(.*?)':\s*(.*)", line)
                if m:
                    requested_label = m.group(1)
                    out_text = m.group(2)
                    parsed = parse_bbox(out_text, width, height)
                    if parsed:
                        # Pick the largest box to filter out tiny noisy boxes
                        best_box = max(parsed, key=lambda item: (item["box"][2]-item["box"][0])*(item["box"][3]-item["box"][1]))["box"]
                        last_raw_boxes[requested_label] = best_box
                        frame_to_raw_boxes[current_frame][requested_label] = best_box
            elif "Skipping inference for frame" in line:
                m = re.search(r'Skipping inference for frame (\d+)', line)
                if m:
                    current_frame = int(m.group(1))
                    frame_to_raw_boxes[current_frame] = dict(last_raw_boxes)

    if not frame_to_raw_boxes:
        print("No processed frames found in the log.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("Applying Exponential Moving Average (EMA) smoothing and redrawing frames...")
    
    tracked_boxes = {}
    frame_idx = 1
    max_frame = max(frame_to_raw_boxes.keys())
    
    # 2. Iterate through the video, apply math smoothing, and redraw
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > max_frame:
            break
            
        raw_boxes = frame_to_raw_boxes.get(frame_idx, {})
        current_labels = set(raw_boxes.keys())
        
        # Apply EMA Math
        for label, box in raw_boxes.items():
            if label in tracked_boxes:
                tx1, ty1, tx2, ty2 = tracked_boxes[label]
                nx1, ny1, nx2, ny2 = box
                tracked_boxes[label] = (
                    int(alpha * nx1 + (1 - alpha) * tx1),
                    int(alpha * ny1 + (1 - alpha) * ty1),
                    int(alpha * nx2 + (1 - alpha) * tx2),
                    int(alpha * ny2 + (1 - alpha) * ty2)
                )
            else:
                tracked_boxes[label] = box
                
        # Fade out labels that disappeared
        for label in list(tracked_boxes.keys()):
            if label not in current_labels:
                del tracked_boxes[label]
                
        # Draw the perfectly smoothed boxes!
        for label, (x1, y1, x2, y2) in tracked_boxes.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg"), frame)
        if frame_idx % 50 == 0:
            print(f"Redrew {frame_idx} smoothed frames...")
            
        frame_idx += 1
        
    cap.release()
    
    output_path = "output_located_smooth.mp4"
    print(f"\nStitching {frame_idx-1} smoothed frames into final MP4 video...")
    ffmpeg_result = os.system(f"ffmpeg -y -framerate {fps} -i {output_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}")
    
    if ffmpeg_result != 0:
        print("Falling back to OpenCV stitcher...")
        frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
        if frame_files:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for f in frame_files:
                writer.write(cv2.imread(f))
            writer.release()
            
    print(f"\nFinished! Super smooth video saved to {output_path}")

if __name__ == '__main__':
    redraw_smooth()
