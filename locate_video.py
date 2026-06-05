import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import transformers.modeling_utils
import argparse
import re
import os
import glob
from functools import wraps

# --- MONKEYPATCH to fix HuggingFace API compatibility with custom remote code ---
original_init = transformers.modeling_utils.PreTrainedModel.__init__

@wraps(original_init)
def patched_init(self, config, *args, **kwargs):
    # Fix missing rope_theta in config which causes Qwen2 to crash
    if not hasattr(config, "rope_theta"):
        config.rope_theta = 1000000.0
    if hasattr(config, "text_config") and not hasattr(config.text_config, "rope_theta"):
        config.text_config.rope_theta = 1000000.0

    original_method = getattr(self.__class__, "_check_and_adjust_attn_implementation", None)
    
    if original_method:
        def wrapper(self_instance, *a, **kw):
            kw.pop("allow_all_kernels", None)
            return original_method(self_instance, *a, **kw)
            
        # Temporarily bind the wrapper to the instance
        self._check_and_adjust_attn_implementation = wrapper.__get__(self)
        
    try:
        original_init(self, config, *args, **kwargs)
    finally:
        if hasattr(self, "_check_and_adjust_attn_implementation"):
            del self._check_and_adjust_attn_implementation
            
    # Fix for missing all_tied_weights_keys if custom model didn't call post_init()
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}

transformers.modeling_utils.PreTrainedModel.__init__ = patched_init

original_get_expanded = transformers.modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

@wraps(original_get_expanded)
def patched_get_expanded(self, *args, **kwargs):
    try:
        return original_get_expanded(self, *args, **kwargs)
    except AttributeError as e:
        if "'list' object has no attribute 'keys'" in str(e):
            return getattr(self, "_tied_weights_keys", [])
        raise

transformers.modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = patched_get_expanded

# Fix for to_legacy_cache / from_legacy_cache being removed in transformers 4.46+
import transformers.cache_utils
DynCache = transformers.cache_utils.DynamicCache

def _to_legacy_cache(self):
    return self

def _from_legacy_cache(cls, past_key_values=None):
    if isinstance(past_key_values, DynCache):
        return past_key_values
    if past_key_values is None:
        return cls()
    cache = cls()
    for layer_past in past_key_values:
        key_states, value_states = layer_past[:2]
        cache.update(key_states, value_states, len(cache))
    return cache

# Force-set both methods unconditionally
DynCache.to_legacy_cache = _to_legacy_cache
DynCache.from_legacy_cache = classmethod(_from_legacy_cache)
# -------------------------------------------------------------------------------

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
    
    model_id = "nvidia/LocateAnything-3B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=dtype
    ).to(device)
    model.eval()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup crash-proof frame saving
    frames_dir = "output_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Check if we can resume
    existing_frames = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
    start_frame = len(existing_frames)
    
    if start_frame > 0:
        print(f"Found {start_frame} existing frames. Resuming from frame {start_frame + 1}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    last_boxes = []
    last_text = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Stop processing after 30 seconds of video
        if frame_count > fps * 30:
            print(f"Reached 30 seconds of video. Stopping early.")
            break
            
        # Only run the heavy inference every 3rd frame (or the very first frame)
        if frame_count % 3 == 1:
            print(f"Processing frame {frame_count} with LocateAnything...")
            
            # Convert OpenCV BGR frame to PIL RGB Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Format prompt using the chat template required by LocateAnything
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Run inference
            inputs = processor(images=[pil_image], text=text_prompt, return_tensors="pt").to(device)
            
            # Ensure images match the model's floating point precision
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
                
            with torch.no_grad():
                # Generate the bounding boxes/text
                outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, tokenizer=processor.tokenizer)
                
            # Extract the generated text — handle both string and tensor outputs
            output = outputs[0]
            if isinstance(output, str):
                last_text = output
            else:
                if hasattr(output, 'cpu'):
                    output = output.cpu().tolist()
                last_text = processor.decode(output, skip_special_tokens=True)
                
            print(f"Output: {last_text}")
            last_boxes = parse_bbox(last_text, width, height)
        else:
            print(f"Skipping inference for frame {frame_count}, reusing previous boxes.")
            
        # Draw the cached (or newly generated) boxes
        for (x1, y1, x2, y2) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
        # Print the raw text output onto the video frame
        cv2.putText(frame, last_text[:100], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save frame to disk safely
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
    cap.release()
    
    print("\nStitching frames into final MP4 video using FFmpeg...")
    # Compile the frames into the final video
    os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}")
    
    print(f"\nFinished! Processed video saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LocateAnything Video Processing")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_located.mp4', help='Path to output video')
    parser.add_argument('--prompt', type=str, default='Locate the gloved hand, the needle holder, the surgical thread, and the metal cheek retractor.', help='Text prompt for the model to locate')
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.prompt)
