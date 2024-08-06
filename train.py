import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def train_hosts():
    # Load the data
    host_data = pd.read_csv('hosts.csv')
    
    # Combine data into a single text column
    host_data['text'] = host_data.apply(
        lambda row: f"Question: This host state is {row['state']}. Which means the host is {row['event']}. Why this host is {row['event'].split('_')[1]}? Answer: {row['reason']}",
        axis=1
    )
    
    # Load the tokenizer and add a padding token
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token
    
    # Tokenize the text data
    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens
    
    # Create dataset
    dataset = Dataset.from_pandas(host_data[['text']])
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.resize_token_embeddings(len(tokenizer))  # Adjust the token embeddings to match the tokenizer size
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Increase the number of epochs
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    # Train the model
    trainer.train()
    
    # Save the tokenizer and model
    tokenizer.save_pretrained('./results/checkpoint-9')
    model.save_pretrained('./results/checkpoint-9')

train_hosts()
