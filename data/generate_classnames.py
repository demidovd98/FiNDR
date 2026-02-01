import os

import time
import datetime

import json

from data import DATA_STATS


### Configurable options
API_KEY = "<your_api_key>"
AI_PROVIDER = "qwen"        # specify the AI provider ["openai", "google", "qwen"]
if AI_PROVIDER == "qwen":
    FREE_VERSION = False    # Set to True to use the free version of Qwen model (with limited capabilities)
THINKING = True             # for Gemini Flash only (control thinking budget)
if THINKING:
    # P.S. For qwen and gemini-2.5-flash (for experiments wih 0 and full reasoning):        
    thinking_budget = 24576  # 8192     # 0 - Completely turns off reasoning.
                                        # -1 - Enables dynamic thinking (auto via model).
                                        # 1 ≤ thinkingBudget ≤ 24,576 - Sets a manual cap on reasoning tokens.    
else:
    thinking_budget = 0                 # Use no reasoning for the basic prompt

DATASET_TYPE = "birds"          # options: ["birds", "cars", "dogs", "flowers", "pets"]
MAIN_PROMPT_TYPE = "basic"      # ["basic", "step_by_step"]
PROMPT_DESIGN = "base_meta-category_expert"  # options: ["base", "base_meta-category", "base_meta-category_expert" (main prompt), "base_meta-category_expert_limits", "base_meta-category_expert_limits_dataset-specifics"]
META_INFO_AUTO = True           # Set to True to auto-generate meta information for the prompt

AUTO_RESUME = True              # Set to True to auto_resume in a loop from the last processed image 
                                # and for failed generations (from the saved JSON file)
if AUTO_RESUME:
    # Provide the path here for manual load of the existing JSON file
    json_path_resume = ''  #  empty value if False, just in case
###


### dev zone
PRINT_EVERY_N_IMAGES = 10  # Print every N images processed
DEBUG_DETAILED = False  # Set to True for detailed debug output
###


### Utils
# Function to encode image to base64 (for OpenAI API)
def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to encode image to bytes (for Google GenAI API)
def encode_image_bytes(path):
    with open(path, "rb") as image_file:
        return image_file.read()

# TODO provide real dataset path
def get_image_paths(dataset_path):
    # For each sub-directory in the dataset directory get a list of image paths and merge them into a single list
    image_paths = []
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths
###


def generate_output(input, model_type, request_type, img_id):
    success = False
    for try_id in range(100):

        if ((img_id + (try_id+1)) % (3*10) == 0) or ((try_id+1) % 10 == 0):
            print("[INFO] Sleeping for 20 secs")
            time.sleep(20)

        try:
            if AI_PROVIDER == "google":
                if model_type[:14] == "gemini-2.5-pro":
                    response = client.models.generate_content(
                        model=model_type,
                        contents=input,
                    )
                else:
                    response = client.models.generate_content(
                        model=model_type,
                        contents=input,
                        config=model_main_reasoning,
                    )                    
                response_text = response.text

            elif AI_PROVIDER == "openai":
                ## normal API call (faster, more expensive)
                # response_main = openai.ChatCompletion.create(
                #     model=model_type,
                #     messages=messages_main
                # )
                ## flex API call (cheaper, with delay)
                response = client.responses.create(
                    model=model_type,
                    # instructions="", # can move expert instructions here
                    input=input,
                    reasoning=model_main_reasoning,
                    max_output_tokens=25000,  # recommended > 25,000 for reasoning; default is ?
                    service_tier="flex",  # "auto" or "flex"; default is "auto"
                    timeout=900.0  # for "flex" API increase default timeout to 15 minutes; defualt is 600.0 (10 minutes)
                )
                response_text = response.output_text
            
            elif AI_PROVIDER == "qwen":
                response = client.chat.completions.create(
                    extra_body={},
                    model=model_type,
                    # provider={
                    #             "only": ["hyperbolic", "chutes"]
                    #         },                                       
                    messages=input,
                    # reasoning=model_main_reasoning,
                )
                response_text = str(response.choices[0].message.content)

            if request_type == "service":
                # find dictionary in the response text
                try:
                    response_text = response_text.split("```python")[1]
                    response_text = response_text.split("```")[0]
                except Exception as error:
                    if img_id % PRINT_EVERY_N_IMAGES == 0:
                        print(f"[WARNING] Header ```python``` is not found the output: {error}")

                text_dict = eval(response_text)  # TODO: use a safer loading method
                assert isinstance(text_dict, dict), f"[ERROR] Output was not read as a dict type"
                response_text = text_dict

            success = True
            break
        except Exception as error:
            if img_id % PRINT_EVERY_N_IMAGES == 0:
                print(f"[ERROR] Raw error information: {error}")
                print(f"[ERROR] An error occurred while generating a prompt, retrying for {try_id+1}/{100} attempts")

    if not success:
        if img_id % PRINT_EVERY_N_IMAGES == 0:
            print(f"[ERROR] No response was generated for image {img_id} after {try_id+1} attempts")
        response = None
        response_text = None

    return response, response_text, success
        


## Parse the prompt design options and set the corresponding flags
if MAIN_PROMPT_TYPE == "step_by_step":
    prompt_design_expert = True
elif MAIN_PROMPT_TYPE == "basic":
    if "expert" in PROMPT_DESIGN:
        prompt_design_expert = True
    else:
        prompt_design_expert = False

if "dataset-specifics" in PROMPT_DESIGN:
    use_dataset_specifics = True
else:
    use_dataset_specifics = False

## Prepare dataset specifics
dataset_specifics = {
    "birds": {
        "path": "./datasets/birds_200/images_discovery_all_3/",
        "expert_name": "" if META_INFO_AUTO else "zoologist",
        "category_singular": "" if META_INFO_AUTO else "bird",
        "category_plural": "" if META_INFO_AUTO else "birds",
        "unit_singular": "" if META_INFO_AUTO else "species",
        "unit_plural": "" if META_INFO_AUTO else "species",
        "unit_specifics": "",
        "classname_specifics_main": "",
        "classname_specifics_service": "",  # keep the space at the beginning (if used)
    },
    "cars": {
        "path": "./datasets/cars_196/images_discovery_all_3/",
        "expert_name": "" if META_INFO_AUTO else "car dealer",
        "category_singular": "" if META_INFO_AUTO else "car",
        "category_plural": "" if META_INFO_AUTO else "cars",
        "unit_singular": "" if META_INFO_AUTO else "model",
        "unit_plural": "" if META_INFO_AUTO else  "models",
        "unit_specifics": "" if not use_dataset_specifics else " followed by its specific single production year", 
        "classname_specifics_main": "" if not use_dataset_specifics else "Prefer a car model name typically sold and used in North America (specifically, in the United States and Canada) over a model name used in other countries.",
        "classname_specifics_service": "" if not use_dataset_specifics else " and other details in this order: first use car brand, next use model name, next if the car is hybrid add Hybrid, next if applciable add car type (SUV, Sedan, Cab, Van, Convertible, Coupe, Minivan, Hatchback, Wagon, Cargo Van, etc), lastly add model's production year",
    },
    "dogs": {
        "path": "./datasets/dogs_120/images_discovery_all_3/",
        "expert_name": "" if META_INFO_AUTO else "zoologist",
        "category_singular": "" if META_INFO_AUTO else "dog",
        "category_plural": "" if META_INFO_AUTO else "dogs",
        "unit_singular": "" if META_INFO_AUTO else "breed",
        "unit_plural": "" if META_INFO_AUTO else "breeds",
        "unit_specifics": "",
        "classname_specifics_main": "", 
        "classname_specifics_service": "", # keep the space at the beginning (if used)
    },
    "flowers": {
        "path": "./datasets/flowers_102/images_discovery_all_3/",
        "expert_name": "" if META_INFO_AUTO else "botanist",
        "category_singular": "" if META_INFO_AUTO else "flower",
        "category_plural": "" if META_INFO_AUTO else "flowers",
        "unit_singular": "" if META_INFO_AUTO else "species",
        "unit_plural": "" if META_INFO_AUTO else "species",
        "unit_specifics": "",
        "classname_specifics_main": "" if not use_dataset_specifics else "Prefer a more common flower species name over a scientific name.",
        "classname_specifics_service": "" if not use_dataset_specifics else " (prefer a more common flower species name over a scientific name)",
    },
    "pets": {
        "path": "./datasets/pets_37/images_discovery_all_3/",
        "expert_name": "" if META_INFO_AUTO else "zoologist",
        "category_singular": "" if META_INFO_AUTO else "pet",
        "category_plural": "" if META_INFO_AUTO else "pets",
        "unit_singular": "" if META_INFO_AUTO else "breed",
        "unit_plural": "" if META_INFO_AUTO else "breeds",
        "unit_specifics": "",
        "classname_specifics_main": "", 
        "classname_specifics_service": "",  # keep the space at the beginning (if used)
    },
}


## Initialize the AI client
if AI_PROVIDER == "google":
    from google import genai
    from google.genai import types
    # from google.genai.types import GenerationConfig, ThinkingConfig

    model_main = "gemini-2.5-flash-preview-05-20"  # options: ["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-06-05" #"gemini-2.5-pro-preview-05-06"  #'gemini-2.5-pro-exp-03-25', #'gemini-2.5-pro-preview-03-25']  # Specify the image+text reasoning model
    model_service = "gemini-2.5-flash-preview-05-20"  # options: ["gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-preview-04-17"]  # Specify the text model

    client = genai.Client(api_key=API_KEY)

    # P.S. For gemini-2.5-flash only (for experiments wih 0 and full reasoning):
    model_main_reasoning=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    )

elif AI_PROVIDER == "openai":
    from openai import OpenAI
    import base64

    model_main = "o4-mini"  # options: ["o4-mini", "o3"]  # Specify the image+text reasoning model
    model_service = "gpt-4o-mini"  # options: ["gpt-4o-mini", "o4-mini", "o3"]  #Specify the text model

    model_main_reasoning = {
            "effort": "medium", # "low", "medium", "high"; default is "medium"
            # "summary": "auto", # "detailed", "concise", "auto" or "none"; default is "auto"
            # "max_tokens": thinking_budget,
            }

    client = OpenAI(
        api_key = API_KEY,
        # increase default timeout to 15 minutes (from 10 minutes)
        # timeout=900.0
    )

elif AI_PROVIDER == "qwen":
    from openai import OpenAI
    import base64

    if FREE_VERSION:
        model_main = "qwen/qwen2.5-vl-72b-instruct:free"  # Specify the image+text reasoning model
        model_service = "qwen/qwen2.5-vl-72b-instruct:free"  # Specify the text model
        api_key = API_KEY
    else:
        model_main = "qwen/qwen2.5-vl-72b-instruct"  # Specify the image+text reasoning model
        model_service = "qwen/qwen2.5-vl-72b-instruct"  # Specify the text model
        api_key = API_KEY

    # keyword "reasoning" is not supported by Qwen currently
    model_main_reasoning = {
            "effort": "medium", # options: ["low", "medium", "high"]; default is "medium"
            #"summary": "auto", # options: ["detailed", "concise", "auto" or "none"]; default is "auto"
            "max_tokens": thinking_budget,
            "exclude": True,
            "enabled": True if (thinking_budget > 0) 
                            else False, # True to enable reasoning, False to disable it
            }

    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = api_key,
    )

# Prepare the image paths
image_paths = get_image_paths(dataset_specifics[DATASET_TYPE]['path'])  # Function to get image paths from the dataset



########### Meta-information call
if META_INFO_AUTO:
    prompt_meta_info = str(f"""You are given a set of images representing a specific object category. Analyze these images and provide information about the main object in the images:
1. The category describing these specific objects (sungular and plural forms).
2. The word typically used to describe a unit (or a sub-category) of this category, to distinct such specific similar objects (singular and plural forms).
3. The word typically used to describe an recognized expert or professional who studied this category and is able to easily distinct its units.

Please provide this information in this specific format as a JSON object with the following fields:
{{
    "category_singular": "<category_singular>",
    "category_plural": "<category_plural>",
    "unit_singular": "<unit_singular>",
    "unit_plural": "<unit_plural>",
    "expert_name": "<expert_name>"
}}

Do not provide any additional word or information.
""")

if META_INFO_AUTO:
    print("[INFO] Templates for the meta information prompt:")
    print(f"Expert:  {prompt_meta_info}")

    if len(image_paths) > 3:
        multiple_images = [image_paths[0],  # Add the first image
                            image_paths[(len(image_paths) // 2)],  # Add the middle image
                            image_paths[-1]  # Add the last image
                            ]
    else:
        multiple_images = image_paths
    print(f"[INFO] Processing multiple images: {multiple_images}")

    ## Prepare input data
    # Initialize conversation history
    if (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
        # Initialize messages for OpenAI or Qwen
        messages_main = []

    if AI_PROVIDER == "google":
        content_main = []
        for image_path in multiple_images:
            # Encode the image to bytes
            image_bytes = encode_image_bytes(image_path)

            # Prepare main request content
            content_main.append[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
            ]
        content_main.append(prompt_meta_info)

    elif (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
        content_main = []
        for image_path in multiple_images:
            # Encode the image
            image_base64 = encode_image_base64(image_path)

            # Prepare main request content
            content_main.append(
                {
                    "type": "image_url",
                    "image_url": str(f"data:image/jpeg;base64,{image_base64}"),
                    "detail": "high",
                },       
            )
        content_main.append(
            {
                "type": "text", 
                "text": prompt_meta_info,
            }
        )

    ## request using image and text
    while True:
        print(f"[INFO] Generating meta information for the dataset '{DATASET_TYPE}' using {AI_PROVIDER} provider and model '{model_main}'")
        
        if AI_PROVIDER == "google":
            response_main, response_main_text, success = generate_output(input=content_main, model_type=model_main, request_type="main", img_id=0)

            # Extract and print assistant's response
            assistant_reply_main = response_main_text
            print(f"[INFO] Assistant Main ({model_main}): {assistant_reply_main}")

        elif (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
            # Append first user message
            messages_main.append({
                "role": "user",
                "content": content_main
            })
            response_main, response_main_text, success = generate_output(input=messages_main, model_type=model_main, request_type="main", img_id=0)

            # Extract and print assistant's response
            assistant_reply_main = response_main_text
            print(f"Assistant Main ({model_main}): {assistant_reply_main}")
            print(f"Assistant Main ({model_main}) usage:")
            print(response_main.usage)

        # use JSON object output as a dictionary:
        if success:
            try:
                # Parse the response text as a JSON object
                meta_info = json.loads(assistant_reply_main)
                print(f"[INFO] Meta information: {meta_info}")
            except Exception as error:
                print(f"[ERROR] Failed to parse meta information: {error}")
                continue
        else:
            continue

        for key in ["category_singular", "category_plural", "unit_singular", "unit_plural", "expert_name"]:
            if key not in meta_info:
                print(f"[ERROR] Key '{key}' not found in meta_info: {meta_info}")
                continue
            dataset_specifics[DATASET_TYPE][key] = meta_info[key]

        print(f"[INFO] Meta information successfully updated: {dataset_specifics[DATASET_TYPE]}")
        break



########### Main prediction call

### Prepare input data

## Prepare the expert message
if not prompt_design_expert:
    text_expert = str(f"""""")
else:
    text_expert = str(f"""You are a professional {dataset_specifics[DATASET_TYPE]['expert_name']} and an expert in {dataset_specifics[DATASET_TYPE]['category_singular']} classification.
    """)

## Prepare the main prompt (main prediction)
# step-by-step prompt
if MAIN_PROMPT_TYPE == "step_by_step":
    prompt_main = str(f"""First, always analyze the provided image of a {dataset_specifics[DATASET_TYPE]['category_singular']}.
Next, think about the most distinctive attributes of the {dataset_specifics[DATASET_TYPE]['category_singular']} and its surroundings in this image. These attributes should be valuable and distinctive enough to distinguish this specific {dataset_specifics[DATASET_TYPE]['category_singular']} from other {dataset_specifics[DATASET_TYPE]['unit_plural']} of visually similar {dataset_specifics[DATASET_TYPE]['category_plural']}.
The other {dataset_specifics[DATASET_TYPE]['unit_plural']} of {dataset_specifics[DATASET_TYPE]['category_plural']} can be very visually similar, so the {dataset_specifics[DATASET_TYPE]['category_singular']} details and background can both be important. 
Lastly, think about what similar {dataset_specifics[DATASET_TYPE]['unit_plural']} of {dataset_specifics[DATASET_TYPE]['category_plural']} may fit these attributes and refine the chosen attributes to make them more distinctive.
Write the attributes in the numbered order of importance with 1 as a starting index.

Next, analyze the attributes of a {dataset_specifics[DATASET_TYPE]['category_singular']} and think what could be the top 2-3 very specific {dataset_specifics[DATASET_TYPE]['category_plural']} that match these attributes.
Ensure that for each suggestion only one specific {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']} is suggested, do not suggest {dataset_specifics[DATASET_TYPE]['unit_plural']} combinations or hybrids. 
{dataset_specifics[DATASET_TYPE]['classname_specifics_main']}
Write the suggestions starting from the most matching {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}.
""")

# generic prompt
else:
    if PROMPT_DESIGN == "base":
        ### base:
        prompt_main = str(f"""What is the exact main object in the provided image?""")

    elif PROMPT_DESIGN == "base_meta-category" or PROMPT_DESIGN == "base_meta-category_expert":
        ### base + meta category:
        prompt_main = str(f"""What is the exact {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}{dataset_specifics[DATASET_TYPE]['unit_specifics']} in the provided image?""")

    elif PROMPT_DESIGN == "base_meta-category_expert_limits":
        ### base + meta category + limits:
        prompt_main = str(f"""What is the exact {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}{dataset_specifics[DATASET_TYPE]['unit_specifics']} in the provided image?
Ensure that only one specific {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']} is suggested, do not suggest {dataset_specifics[DATASET_TYPE]['unit_plural']} combinations or hybrids.""")

    elif PROMPT_DESIGN == "base_meta-category_expert_limits_dataset-specifics":
        ### full (base + meta category + limits + dataset specifics):
        prompt_main = str(f"""What is the exact {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}{dataset_specifics[DATASET_TYPE]['unit_specifics']} in the provided image?
Ensure that only one specific {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']} is suggested, do not suggest {dataset_specifics[DATASET_TYPE]['unit_plural']} combinations or hybrids.
{dataset_specifics[DATASET_TYPE]['classname_specifics_main']}""")

    else:
        raise ValueError(f"[ERROR] Unknown prompt design: {PROMPT_DESIGN}. Please choose from ['base', 'base_meta-category', 'base_meta-category_expert', 'base_meta-category_expert_limits', 'base_meta-category_expert_limits_dataset-specifics']")

# Prepare the service prompt (post-processing)
prompt_service = str(f"""Convert the below text containing suggested {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_plural']}{dataset_specifics[DATASET_TYPE]['unit_specifics']} to a Python dictionary object, where a key is an index and the value is a suggestion of the specific {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}{dataset_specifics[DATASET_TYPE]['unit_specifics']}. 
Only use the final {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_singular']}{dataset_specifics[DATASET_TYPE]['unit_specifics']} prediction(s), do not use any intermediate suggestions.
Remove duplicated suggestions and unsepcific {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_plural']}. Also keep the numbered order of the suggestions with 1 as a starting index.
Make sure to only use English letters. Add a space between seprate words if not done in the suggested {dataset_specifics[DATASET_TYPE]['category_singular']} {dataset_specifics[DATASET_TYPE]['unit_plural']} and capitalize abbreviations and first letters of normal words.

The text:
""")

print("[INFO] Templates for the input prompts:")
print(f"Expert:  {text_expert}")
print(f"Main:  {prompt_main}")
print(f"Service:  {prompt_service}")

start_stamp = str(datetime.datetime.now())
run_directory = str(f"./data/guessed_classnames/{DATASET_TYPE[:-1]}{DATA_STATS[DATASET_TYPE[:-1]]['num_classes']}/{DATASET_TYPE}_{AI_PROVIDER}_{start_stamp}")  # Create a unique directory name for the run

# create a directory for the output JSON file if it does not exist
if not os.path.exists(run_directory):
    os.makedirs(run_directory)


### Generation loop
while True:
    if AUTO_RESUME and (json_path_resume != ''):
        print(f"[INFO] Loading existing JSON file from {json_path_resume}")
        with open(json_path_resume, 'r', encoding='utf-8') as f:
            out_json = json.load(f)

        bad_indeces = []
        for key, value in out_json.items():
            if (value['model_main_success'] == False) \
                        or (value['model_service_success'] == False) \
                        or (len(value['guessed_classnames']) == 0):
                bad_indeces.append(value['image_id'])
        print(f"[INFO] Images with bad indeces: {bad_indeces}")

        if (len(bad_indeces) == 0) and (len(out_json.keys()) == len(image_paths)):
            print(f"[INFO] No images with bad indeces found, exiting")
            break

    else:
        bad_indeces = None
        out_json = {}

    now = str(datetime.datetime.now())

    # --- Loop through all images ---
    for img_id, image_path in enumerate(image_paths):

        if img_id % PRINT_EVERY_N_IMAGES == 0:
            print(f"[INFO] Processing image {img_id+1}/{len(image_paths)}: {'/'.join(image_path.split('/')[-2:])}")

        # if "/".join(image_path.split("/")[-2:]) in out_json.keys():
        if AUTO_RESUME and (bad_indeces is not None):
            if img_id not in bad_indeces:
                if DEBUG_DETAILED and (img_id % PRINT_EVERY_N_IMAGES == 0):
                    print(f"[INFO] Image {img_id+1}/{len(image_paths)}: {'/'.join(image_path.split('/')[-2:])} was already processed, skipping and loading results from the JSON file")
                continue


        # -- Prepare input data --

        # Initialize conversation history
        if (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
            # Initialize messages for OpenAI or Qwen
            messages_main = []
            messages_service = []

        if AI_PROVIDER == "google":
            # Encode the image
            image_bytes = encode_image_bytes(image_path)

            # Prepare main request content
            content_main = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                text_expert,
                prompt_main,
            ]

            # Prepare service request content
            content_service = [
                text_expert,
                prompt_service,
            ]

        elif (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
            message_expert = {
                    "role": "developer", 
                    "content": text_expert
                }

            # Encode the image
            image_base64 = encode_image_base64(image_path)

            # Prepare main request content
            content_main = [
                {
                    "type": "text", 
                    "text": prompt_main,
                },
                {
                    "type": "image_url",
                    "image_url": str(f"data:image/jpeg;base64,{image_base64}"),
                    "detail": "high",
                }
            ]

            # Prepare service request content
            content_service = [
                {
                    "type": "text", 
                    "text": prompt_service
                },
            ]


        # -- Main Request: using image and text --

        if AI_PROVIDER == "google":
            response_main, response_main_text, success = generate_output(input=content_main, model_type=model_main, request_type="main", img_id=img_id)

            # Extract and print assistant's response
            assistant_reply_main = response_main_text
            if (img_id) % (3*10) == 0: print(f"[INFO] Assistant Main ({model_main}): {assistant_reply_main}")

        elif (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
            # Start the conversation with the expert message
            messages_main.append(message_expert)

            # Append first user message
            messages_main.append({
                "role": "user",
                "content": content_main
            })

            response_main, response_main_text, success = generate_output(input=messages_main, model_type=model_main, request_type="main", img_id=img_id)

            # Extract and print assistant's response
            assistant_reply_main = response_main_text
            if img_id % PRINT_EVERY_N_IMAGES == 0: 
                print(f"Assistant Main ({model_main}): {assistant_reply_main}")
                print(f"Assistant Main ({model_main}) usage:")
                print(response_main.usage)


        # -- Service Request: Using text only --

        if AI_PROVIDER == "google":
            # Append assistant's response to messages
            content_service.append({
                assistant_reply_main
            })
            response_service, response_service_text, success = generate_output(input=content_service, model_type=model_service, request_type="service", img_id=img_id)

            # Extract and print assistant's response
            assistant_reply_service = response_service_text
            if (img_id) % (3*10) == 0: print(f"[INFO] Assistant Service ({model_service}): {assistant_reply_service}")

        elif (AI_PROVIDER == "openai") or (AI_PROVIDER == "qwen"):
            # Start the conversation with the expert message
            messages_service.append(message_expert)

            # Append assistant's response to messages
            messages_service.append({
                "role": "user",
                "content": content_service
            })

            messages_service.append({
                "role": "user",
                "content": "Ensure to output only a Python dictionary object and nothing else."
            })

            # Append assistant's response to messages
            messages_service.append({
                "role": "assistant", # or betetr "user" here ?
                "content": str(assistant_reply_main)
            })

            # Send request to gpt-4o-mini model
            response_service, response_service_text, success = generate_output(input=messages_service, model_type=model_service, request_type="service", img_id=img_id)

            # Extract and print assistant's response
            assistant_reply_service = response_service_text

            if img_id % PRINT_EVERY_N_IMAGES == 0:
                print(f"Assistant Service ({model_service}): {assistant_reply_service}")
                print(f"Assistant Service ({model_service}) usage:")
                print(response_service.usage)

            # Append assistant's response to messages
            messages_service.append({
                "role": "assistant",
                "content": assistant_reply_service
            })

        classname = image_path.split("/")[-2:]
        classname = '/'.join(classname) # get the image name with class name
        out_json[classname] = {
            "guessed_classnames": assistant_reply_service,

            "image_path": image_path,
            "image_id": img_id,
            
            "model_main": model_main,
            "model_main_response": assistant_reply_main,
            "model_main_success": success,

            "model_service": model_service,
            "model_service_response": assistant_reply_service,
            "model_service_success": success,
        }
        if img_id % (3*10) == 0: # intermediate save just in case (rewrite the file)
            with open(os.path.join(run_directory, str(f'data_{DATASET_TYPE}_{str(now)}.json')), 'w', encoding='utf-8') as f:
                json.dump(out_json, f, ensure_ascii=False, indent=4)   


    # final save
    json_name = str(f'data_{DATASET_TYPE}_{str(now)}.json')
    with open(os.path.join(run_directory,  json_name), 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=4)
    print(f"[INFO] All images were processed, output saved to {run_directory}/{json_name}")

    if AUTO_RESUME:
        json_path_resume = os.path.join(run_directory, json_name)
        print(f"[INFO] Resuming is done, exiting")
    else:
        print(f"[INFO] All images were processed, exiting")
        break