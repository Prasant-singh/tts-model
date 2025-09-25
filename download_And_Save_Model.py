import os
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# --- Configuration ---
model_name = "ai4bharat/indic-parler-tts"
base_local_path = "./indic-parler-tts-local-structured" # Use a new name to avoid confusion

def download_and_save_model_structured():
    """
    Downloads the Parler-TTS model and its tokenizers and saves them
    in a structured local directory with subfolders to prevent conflicts.
    """
    print(f"Saving model '{model_name}' to '{base_local_path}' with structured directories...")

    try:
        # --- Download the main model ---
        print("Downloading main Parler-TTS model...")
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)

        # --- Download the primary (prompt) tokenizer ---
        print("Downloading primary tokenizer...")
        prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- Download the description tokenizer ---
        print("Downloading description tokenizer...")
        description_tokenizer_name = model.config.text_encoder._name_or_path
        description_tokenizer = AutoTokenizer.from_pretrained(description_tokenizer_name)

        # --- Define paths for each component ---
        model_path = os.path.join(base_local_path, "model")
        prompt_tokenizer_path = os.path.join(base_local_path, "prompt_tokenizer")
        description_tokenizer_path = os.path.join(base_local_path, "description_tokenizer")

        # --- Save each component to its dedicated subdirectory ---
        print(f"Saving main model to '{model_path}'...")
        model.save_pretrained(model_path)

        print(f"Saving prompt tokenizer to '{prompt_tokenizer_path}'...")
        prompt_tokenizer.save_pretrained(prompt_tokenizer_path)
        
        print(f"Saving description tokenizer to '{description_tokenizer_path}'...")
        description_tokenizer.save_pretrained(description_tokenizer_path)

        print("\nAll components downloaded and saved successfully in a structured format!")

    except Exception as e:
        print(f"\nAn error occurred during download: {e}")

if __name__ == "__main__":
    download_and_save_model_structured()