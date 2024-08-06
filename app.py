import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate a response based on the given prompt using a pre-trained GPT-2 model.

    Args:
        prompt (str): The input prompt to generate a response for.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature for response generation.
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): Cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.

    Returns:
        str: The generated response.
    """
    try:
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Generate response
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response
    
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return "Sorry, I couldn't generate a response."

if __name__ == "__main__":
    prompt = "Explain predicate logic in terms of propositional logic."
    print(generate_response(prompt))
