import json
import os

VISION_URL = "https://api.openai.com/v1/chat/completions"


def make_headers(api_key): 
    headers = { 
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    return headers


def make_payload(img_b64, img_path, text="Classify this image into one of four categories: flooded road, wet road, dry road, no road shown.", model="gpt-4-vision-preview", max_tokens=300):

    text += f"Return the result in JSON format, with key {os.path.splitext(os.path.basename(img_path))[0]} and value as the classification."
     
    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": int(f"{max_tokens}")
    }
    return payload
