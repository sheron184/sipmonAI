from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./results/checkpoint-15')
tokenizer.pad_token = tokenizer.eos_token  # Ensure the padding token is set correctly

# Load the model
model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-15')

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Question: This host state is New York. Which means the host is an event_host. Why this host is event?"
generated_text = generate_text(prompt)
print(generated_text)
