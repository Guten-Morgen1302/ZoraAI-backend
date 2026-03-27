import ollama

def chat_with_phi3(prompt):
    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    )
    return response['message']['content']

# Example usage
user_prompt = "Explain quantum computing simply."
result = chat_with_phi3(user_prompt)
print(result)
