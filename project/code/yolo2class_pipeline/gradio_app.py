import gradio as gr
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tempfile


API_URL = "http://localhost:8000/inference"

def process_image(image):
    try:
        img_byte_arr = io.BytesIO()

        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(API_URL, files=files)
        
        if response.status_code != 200:
            return image, f"Error: {response.text}"
        
        result = response.json()
        
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Create a font object with larger size
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)
        except:
            font = ImageFont.load_default()  # Fallback to default font
        
        for box, label, det_conf, cls_conf in zip(
            result['boxes'],
            result['labels'],
            result['detector_confidences'],
            result['classifier_confidences']
        ):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            label_text = f"{label}"
            # Draw text with the larger font
            draw.text((x1, y1-40), label_text, fill='red', font=font)
        
        return draw_image, "Processing complete"
    except Exception as e:
        return image, f"Error processing image: {str(e)}"

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=[
        gr.Image(type="pil"),
        gr.Textbox()
    ],
    title="FlyDetector",
    description="Upload an image (PNG or JPG) to classify flies. The app will return the image with bounding boxes and classification results.",
    examples=[],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860) 