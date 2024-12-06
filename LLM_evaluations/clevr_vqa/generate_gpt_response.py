import base64
import requests
import os
import json
import numpy as np
from tqdm import tqdm

model = 'gpt-4-vision-preview'

# replace this with your own OpenAI API key!
API_KEY=None
assert API_KEY is not None

np.random.seed(0)

num_samples = 500

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(image_path, question):
    # OpenAI API Key (my personal key)
    api_key = API_KEY

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{question} First answer either rubber, metal, cube, sphere, cylinder, gray, red, blue, green, brown, purple, cyan, yellow, left, right, behind, front, small, large, true, false or an integer with no additional content, and then explain your answer in separate sentences."
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response


image_dir = "../../benchmark_data/clevr_vqa/images"

# Load CLEVR_questions.json file
questions_json_path = "../../benchmark_data/clevr_vqa/CLEVR_questions.json"
questions_dict = json.load(open(questions_json_path, 'r')) 
questions = questions_dict['questions']

# Create Dictionary of Image Names and Question/Answer Pairs
image_qa_pairs = {}
for question in questions:
    if question['image_filename'] not in image_qa_pairs:
        image_qa_pairs[question['image_filename']] = []
    image_qa_pairs[question['image_filename']].append({'question': question['question'], 'answer': question['answer']})

gpt_answers = {}
output_folder = "./gpt4_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with tqdm(total=num_samples, desc="Processing images", unit="file", ncols=100) as pbar:
  for filename in sorted([file for file in os.listdir(image_dir) if file.endswith('.png')])[:num_samples]:
        selected_question = np.random.choice(image_qa_pairs[filename])
    
        result_fn = os.path.join(output_folder, filename.split('.')[0] + '_gpt.json')
        if os.path.exists(result_fn):
            print('skip because already processed: ', result_fn)
            continue

        image_path = os.path.join(image_dir, filename)
        response = get_response(image_path, selected_question['question'])
        #print(response.json())
        answer = response.json()["choices"][0]["message"]['content']
        #print(answer)
        gpt_answers[filename] = {
            'Asked Question': selected_question['question'],
            'Correct Answer': selected_question['answer'],
            'GPT Answer': answer,
            'short_answer': (answer.split('\n')[0])
        }
        #print(gpt_answers[filename])
       
        with open(result_fn, 'w') as f:
          json.dump(gpt_answers[filename], f, indent=4)
        #print(filename)
        pbar.update(1)
