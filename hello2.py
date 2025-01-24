#!/usr/bin/env -S uv run -q

# Instructions:
# 1. Have an Apple Silicon Mac
# 2. Install Python from https://python.org
# 3. Create a project directory
# 4. Create a virtual environment: python3 -m venv venv
# 5. Activate the virtual environment: source venv/bin/activate
# 6. Install the mlx library: pip install mlx
# 7. Edit the prompt variable to taste.
# 8. Run the script: python3 talk.py
#
# Optional: 
# 1. Install lmstudio.ai for debugging.
# 2. Increase max_tokens if you need longer responses.

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

prompt = "How can I become healthier? Give me 10 of your best tips."

# Sets the maximum number of response tokens.
# Default is 256, I think.
# The number of tokens is dumped to the console when verbose=True in generate()
#     Generation: 513 tokens, 123.774 tokens-per-sec 
max_tokens = 2048

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)
