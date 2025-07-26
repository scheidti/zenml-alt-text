generate_alt_text_prompt = """You are an expert in web accessibility and inclusive design.
Given the provided image, generate concise, descriptive alternative text suitable for use on a webpage.
The alternative text should clearly communicate the essential visual information to users who cannot see the image, such as screen reader users.
Follow best practices in web accessibility, including brevity (ideally fewer than 150 characters), relevance, and context-awareness.
Avoid redundant phrases like 'image of' or 'picture of' unless necessary for clarity.
Provide the alternative text directly without any additional explanation."""

system_message = """You are an expert in web accessibility and inclusive design, specializing in writing clear, meaningful, and descriptive alternative texts for images on webpages."""

user_prompt = """Given the provided image, generate concise and context-aware alternative text (ideally fewer than 150 characters) suitable for screen reader users.
Avoid phrases like "image of" unless essential. Provide only the alt text without explanations.
"""


def format_data_for_training(sample, image_row = "image", alt_text_row = "alt_text"):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image",
                        "image": sample[image_row],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample[alt_text_row]}],
            },
        ],
    }
