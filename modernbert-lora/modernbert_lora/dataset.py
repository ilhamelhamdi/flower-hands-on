from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def get_preprocessed_data(model_id, dataset_config):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    raw_dataset = load_dataset(dataset_config["name"], dataset_config["subset"])

    # 1. Handle Splits (Standardizing names)
    train_data = raw_dataset["train"]
    # Some uses 'test' as validation; GLUE uses 'validation'
    val_key = "validation" if "validation" in raw_dataset else "test"
    val_data = raw_dataset[val_key]

    # 2. Optional Stratified Sampling
    if dataset_config.get("train_size"):
        train_data = train_data.train_test_split(
            test_size=dataset_config["train_size"], 
            stratify_by_column="label", 
            seed=42
        )["test"]

    if dataset_config.get("val_size"):
        val_data = val_data.train_test_split(
            test_size=dataset_config["val_size"], 
            stratify_by_column="label", 
            seed=42
        )["test"]
    
    def encode(examples):
        # Dynamically handle single or pair-sentence tasks
        inputs = [examples[field] for field in dataset_config["text_fields"]]
        return tokenizer(*inputs, truncation=True)

    train_data = train_data.map(encode, batched=True)
    val_data = val_data.map(encode, batched=True)

    train_data = train_data.map(lambda x: {"labels": x["label"]}, batched=True)
    val_data = val_data.map(lambda x: {"labels": x["label"]}, batched=True)
    
    return train_data, val_data, tokenizer, DataCollatorWithPadding(tokenizer=tokenizer)