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

# Only patch get_expanded_tied_weights_keys if it exists (newer transformers)
if hasattr(transformers.modeling_utils.PreTrainedModel, 'get_expanded_tied_weights_keys'):
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

# Only patch DynamicCache if to_legacy_cache is missing (newer transformers removed it)
import transformers.cache_utils
DynCache = transformers.cache_utils.DynamicCache
if not hasattr(DynCache, 'to_legacy_cache'):
    def _to_legacy_cache(self):
        legacy_cache = ()
        for i, layer in enumerate(self.layers):
            k, v = layer.keys, layer.values
            legacy_cache += ((k, v),)
        return legacy_cache

    def _from_legacy_cache(cls, past_key_values=None):
        if isinstance(past_key_values, DynCache):
            return past_key_values
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, layer_past in enumerate(past_key_values):
            cache.update(layer_past[0], layer_past[1], layer_idx)
        return cache

    DynCache.to_legacy_cache = _to_legacy_cache
    DynCache.from_legacy_cache = classmethod(_from_legacy_cache)
# -------------------------------------------------------------------------------

def parse_bbox(text, width, height):
    """
    Attempt to parse grounding bounding boxes and their labels from the model's text output.
    Format: <ref>label</ref><box><x1><y1><x2><y2></box>
    """
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
                # Normalize from 1000-scale to image pixels
                px1 = int((x1 / 1000.0) * width)
                py1 = int((y1 / 1000.0) * height)
                px2 = int((x2 / 1000.0) * width)
                py2 = int((y2 / 1000.0) * height)
                results.append({"label": label.strip(), "box": (px1, py1, px2, py2)})
                
    return results

def process_video(input_path, output_path, prompts):
    print("Loading LocateAnything-3B model... This might take a few moments.")
    
    # Automatically use GPU and fp16 precision if available to save VRAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "nvidia/LocateAnything-3B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=dtype,
        attn_implementation="sdpa"
    ).to(device)
    model.eval()
    
    # Nvidia's custom Qwen2 doesn't support 'eager' attention - force sdpa on all submodules
    for module in model.modules():
        if getattr(module, '_attn_implementation', None) == 'eager':
            module._attn_implementation = 'sdpa'
    print("Attention implementation forced to sdpa.", flush=True)
    
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
            print(f"Reached 30 seconds of video. Stopping early.", flush=True)
            break
            
        # Only run the heavy inference every 3rd frame (or the very first frame)
        if frame_count % 3 == 1:
            print(f"Processing frame {frame_count} with LocateAnything...", flush=True)
            
            # Convert OpenCV BGR frame to PIL RGB Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            all_boxes = []
            all_text = []
            
            for label, desc in prompts.items():
                # Format prompt using the chat template required by LocateAnything
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Locate {desc}. If it is not clearly visible in this frame, output <box>None</box>."}
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
                if isinstance(outputs, str):
                    out_text = outputs
                elif isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str):
                    out_text = outputs[0]
                else:
                    output = outputs[0]
                    if hasattr(output, 'cpu'):
                        output = output.cpu().tolist()
                    out_text = processor.decode(output, skip_special_tokens=False)
                    
                print(f"Output for '{label}': {out_text}")
                all_text.append(out_text)
                
                # Parse boxes and override the label with our exact query for clarity
                parsed = parse_bbox(out_text, width, height)
                for item in parsed:
                    item["label"] = label
                all_boxes.extend(parsed)
                
            last_text = " | ".join(all_text)
            last_boxes = all_boxes
        else:
            print(f"Skipping inference for frame {frame_count}, reusing previous boxes.")
            
        # Draw the cached (or newly generated) boxes
        for item in last_boxes:
            x1, y1, x2, y2 = item["box"]
            label = item["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label background and text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        # Red debug text removed to keep video clean
        
        # Save frame to disk safely
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
    cap.release()
    
    print("\nStitching frames into final MP4 video...", flush=True)
    # Try ffmpeg first, fall back to OpenCV if not installed
    ffmpeg_result = os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}")
    
    if ffmpeg_result != 0:
        print("FFmpeg not found, installing and retrying...", flush=True)
        os.system("sudo apt-get install -y ffmpeg")
        ffmpeg_result = os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}")
    
    if ffmpeg_result != 0:
        print("FFmpeg failed. Falling back to OpenCV stitcher...", flush=True)
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        if frame_files:
            sample = cv2.imread(frame_files[0])
            h, w = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for f in frame_files:
                writer.write(cv2.imread(f))
            writer.release()
    
    print(f"\nFinished! Processed video saved to {output_path}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LocateAnything Video Processing")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_located.mp4', help='Path to output video')
    args = parser.parse_args()
    
    process_video(args.input, args.output)
