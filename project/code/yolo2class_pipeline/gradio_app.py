import gradio as gr
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tempfile
import cv2
import time

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def process_frame(frame):
    try:
        image = Image.fromarray(frame)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/inference", files=files)
        
        if response.status_code != 200:
            return frame, f"Error: {response.text}"
        
        result = response.json()
        
        draw_frame = frame.copy()
        
        for box, label, det_conf, cls_conf in zip(
            result['boxes'],
            result['labels'],
            result['detector_confidences'],
            result['classifier_confidences']
        ):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label_text = f"{label}"
            cv2.putText(draw_frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return draw_frame, "Processing complete"
    except Exception as e:
        return frame, f"Error processing frame: {str(e)}"

def process_image(image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/inference", files=files)
        
        if response.status_code != 200:
            return image, f"Error: {response.text}"
        
        result = response.json()
        
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        try:
            font = ImageFont.truetype("fonts/font.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        for box, label, det_conf, cls_conf in zip(
            result['boxes'],
            result['labels'],
            result['detector_confidences'],
            result['classifier_confidences']
        ):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            label_text = f"{label}"
            draw.text((x1, y1-40), label_text, fill='red', font=font)
        
        return draw_image, "Processing complete"
    except Exception as e:
        return image, f"Error processing image: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# FliesDetector")
    gr.Markdown("Upload an image or use your camera to detect and classify flies in real-time.")
    
    with gr.Tabs():
        with gr.TabItem("Camera"):
            camera_input = gr.Image(sources="webcam", streaming=True)
            camera_output = gr.Image()
            camera_status = gr.Textbox(label="Status")
            camera_input.stream(process_frame, inputs=[camera_input], outputs=[camera_output, camera_status])
            
        with gr.TabItem("Image Upload"):
            image_input = gr.Image(type="pil", image_mode="RGB")
            image_output = gr.Image(type="pil")
            image_status = gr.Textbox(label="Status")
            image_input.change(process_image, inputs=[image_input], outputs=[image_output, image_status])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860) 