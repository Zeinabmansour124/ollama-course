

import requests
import json

url="http://localhost:11434/api/generate"

data = {
    "model": "Mistral",
    "prompt": "Write a short story about a dragon who learns to play the piano."
}

response = requests.post(url, json=data, stream=True)
#check the resonse 
if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            response_json = json.loads(decoded_line)

            generated_text = response_json.get("response", "")
            print(generated_text ,end="" , flush=True)
else:
    print(f"Request failed with status code: {response.status_code}")

