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
            if x1 == 0 and y1 == 0 and x2 >= 990 and y2 >= 990:
                continue
            if x1 <= 1000 and y1 <= 1000 and x2 <= 1000 and y2 <= 1000:
                px1 = int((x1 / 1000.0) * width)
                py1 = int((y1 / 1000.0) * height)
                px2 = int((x2 / 1000.0) * width)
                py2 = int((y2 / 1000.0) * height)
                results.append({"label": label.strip(), "box": (px1, py1, px2, py2)})
    return results

def redraw():
    input_video = "surgery_video.mp4"
    log_file = "processing.log"
    output_dir = "output_frames"
    
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
    frame_to_boxes = {}
    last_boxes = []
    current_frame = 0
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "Processing frame" in line:
                m = re.search(r'Processing frame (\d+)', line)
                if m:
                    current_frame = int(m.group(1))
            elif line.startswith('Output: '):
                text = line[8:]
                last_boxes = parse_bbox(text, width, height)
                frame_to_boxes[current_frame] = last_boxes
            elif "Skipping inference for frame" in line:
                m = re.search(r'Skipping inference for frame (\d+)', line)
                if m:
                    current_frame = int(m.group(1))
                    frame_to_boxes[current_frame] = last_boxes

    os.makedirs(output_dir, exist_ok=True)
    
    print("Redrawing frames...")
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        boxes_to_draw = frame_to_boxes.get(frame_idx, [])
        
        for item in boxes_to_draw:
            x1, y1, x2, y2 = item["box"]
            label = item["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg"), frame)
        if frame_idx % 50 == 0:
            print(f"Redrew {frame_idx} frames...")
        frame_idx += 1
        
    cap.release()
    
    output_path = "output_located_redrawn.mp4"
    print("\nStitching frames into final MP4 video...")
    ffmpeg_result = os.system(f"ffmpeg -y -framerate {fps} -i {output_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}")
    
    if ffmpeg_result != 0:
        print("FFmpeg not found. Falling back to OpenCV stitcher...")
        frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
        if frame_files:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for f in frame_files:
                writer.write(cv2.imread(f))
            writer.release()
            
    print(f"\nFinished! Processed video saved to {output_path}")

if __name__ == '__main__':
    redraw()
