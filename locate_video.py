import cv2
import torch
from PIL import Image
from transformers import pipeline
import argparse
import re

def parse_bbox(text, width, height):
    """
    Attempt to parse grounding bounding boxes from the model's text output.
    Assuming the model outputs coordinates in a normalized [0, 1000] format like [x1, y1, x2, y2].
    """
    boxes = []
    # Matches patterns like [120, 340, 500, 600]
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        # Normalize from 1000-scale to image pixels
        if x1 <= 1000 and y1 <= 1000 and x2 <= 1000 and y2 <= 1000:
            x1 = int((x1 / 1000.0) * width)
            y1 = int((y1 / 1000.0) * height)
            x2 = int((x2 / 1000.0) * width)
            y2 = int((y2 / 1000.0) * height)
            boxes.append((x1, y1, x2, y2))
    return boxes

def process_video(input_path, output_path, prompt):
    print("Loading LocateAnything-3B model... This might take a few moments.")
    
    # Automatically use GPU and fp16 precision if available to save VRAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = pipeline(
        "image-text-to-text", 
        model="nvidia/LocateAnything-3B", 
        trust_remote_code=True, 
        device=device,
        torch_dtype=dtype
    )
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    last_boxes = []
    last_text = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Only run the heavy inference every 3rd frame (or the very first frame)
        if frame_count % 3 == 1:
            print(f"Processing frame {frame_count} with LocateAnything...")
            
            # Convert OpenCV BGR frame to PIL RGB Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
            
            # Run inference
            outputs = pipe(text=messages)
            
            # Extract the generated text
            if isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
                last_text = outputs[0]['generated_text']
            else:
                last_text = str(outputs)
                
            print(f"Output: {last_text}")
            last_boxes = parse_bbox(last_text, width, height)
        else:
            print(f"Skipping inference for frame {frame_count}, reusing previous boxes.")
            
        # Draw the cached (or newly generated) boxes
        for (x1, y1, x2, y2) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
        # Print the raw text output onto the video frame
        cv2.putText(frame, last_text[:100], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"\nFinished! Processed video saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_located.mp4', help='Path to output video')
    parser.add_argument('--prompt', type=str, default='Locate the needle holder, the cheek retractor, and the forceps.', help='Text prompt for the model to locate')
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.prompt)
