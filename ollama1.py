import ollama1

# Send a prompt to your local Llama 3 model
response = ollama1.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Explain quantum computing in one simple sentence.',
  }
])

# Print the model's reply
print(response['message']['content'])