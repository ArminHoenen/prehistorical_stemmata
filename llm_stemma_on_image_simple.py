# code A. Hoenen, 2025, CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/
import openai
import os
import base64

# Set your OpenAI API key
client = openai.OpenAI(api_key="")

# Folder containing PNGs
image_folder = "llm_image_test/"

# Get sorted list of PNG files
png_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])

# Iterate over images
for img_file in png_files:
    print(f"Processing {img_file}")
    img_path = os.path.join(image_folder, img_file)

    # Encode image to base64
    with open(img_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Call model
    completion = client.chat.completions.create(
        model="", #an openAI model you have access to, for instance gpt-4o
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is there a stemma codicum on the image? Only answer with 'yes' or 'no'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=10
    )

    # Get response
    result = completion.choices[0].message.content.strip()

    # Save output
    with open("output.txt", "a") as f:
        f.write(f"{img_file}: {result}\n")
