import gradio as gr
import numpy as np
from PIL import Image
from openai import OpenAI
import os
import base64
import io

 
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE'))

MODELS = [
    "google/gemini-flash-1.5-exp",
    "google/gemini-flash-1.5",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/pixtral-12b:free"
]

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_label(image, model):
    if image is None:
        return "Please upload an image of a product label, food item, or menu."

    image.thumbnail((1024, 1024), Image.LANCZOS)
    base64_image = encode_image(image)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image of a product label, food item, or menu. Provide a simple description of the product and its nutritional information, (for images of food / fruits / dishes / always give approximate calorie values in number with disclaimer that it's generic and it may vary), consider user as if talking to someone with no nutritional background."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1000,
            temperature=0.01
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Define example images
example_images = [
    'images (1).png',
    'images (2).png',
    'images (3).png',
    'images (4).png',
    'images (5).png',
    'images (6).png',
    'images (7).png',
    'images (8).png',
    'images (9).png',
    'images (10).png',
    'images (11).png',
    'images (12).png'
]

iface = gr.Interface(
    fn=analyze_label,
    inputs=[
        gr.Image(type="pil", label="Upload a picture of a product label, food item, or menu"),
        gr.Dropdown(choices=MODELS, label="Select Model", value=MODELS[0])
    ],
    outputs=gr.Textbox(label="Analysis"),
    title="🍏 Sehat GPT 🥦",
    description="Sehat GPT: Your AI nutritionist, inspired by the Label Padhega India campaign. 🤖🥗 Upload food images for instant nutrition insights and make informed dietary choices. Empowering you to read labels, understand nutrition, and lead a healthier life! 💪🍎 Join the movement to promote food literacy and consumer awareness across India. 🇮🇳",
    examples=[[img, MODELS[0]] for img in example_images]
)

# For Gradio spaces, we use this to launch the interface
iface.launch()