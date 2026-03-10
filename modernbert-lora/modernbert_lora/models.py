import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

def get_model(model_id, lora_config_dict, num_labels):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )
        
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        target_modules=lora_config_dict["target_modules"],
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config)
    return model