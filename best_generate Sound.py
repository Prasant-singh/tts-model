# import torch
# from transformers import AutoTokenizer
# import soundfile as sf
# import os
# import datetime

# try:
#     from parler_tts import ParlerTTSForConditionalGeneration
# except ImportError:
#     # ... (rest of the import check is the same)
#     exit()

# def generate_hindi_tts_from_local_structured():
#     """
#     Generates Hindi speech using a locally stored and structured Parler-TTS model.
#     """
#     try:
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {device}")

#         # --- Define the base path for the structured local model ---
#         base_local_path = "./indic-parler-tts-local-structured"
        
#         # --- Define the specific paths for each component ---
#         model_path = os.path.join(base_local_path, "model")
#         prompt_tokenizer_path = os.path.join(base_local_path, "prompt_tokenizer")
#         description_tokenizer_path = os.path.join(base_local_path, "description_tokenizer")

#         print(f"Loading components from base path '{base_local_path}'...")
#         if not os.path.exists(base_local_path):
#             print("Error: Local model directory not found. Please run the corrected download script first.")
#             return

#         # --- Load each component from its dedicated subdirectory ---
#         model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer_path)
#         description_tokenizer = AutoTokenizer.from_pretrained(description_tokenizer_path)
#         print("Model and tokenizers loaded successfully from local structured storage.")

#         prompt = "नमस्ते! मेरा नाम अन्वी है। मैं एक तकनीकी उत्साही हूँ और मुझे पाइथन प्रोजेक्ट्स पर काम करना बेहद पसंद है। खासकर वे प्रोजेक्ट्स जो मशीन लर्निंग, डेटा एनालिटिक्स और कोड सुरक्षा से जुड़े होते हैं। जब भी मुझे कोई नया टूल या लाइब्रेरी मिलती है, तो मैं उसे एक्सप्लोर करने के लिए बहुत उत्साहित हो जाती हूँ। मुझे चुनौतियाँ पसंद हैं—चाहे वह मॉडल को ट्रेन करना हो, या किसी कोड को सुरक्षित बनाना। मैं अक्सर अपने समय का उपयोग नए कौशल सीखने, वर्कफ़्लो को बेहतर बनाने और अपने प्रोजेक्ट्स को अधिक प्रभावशाली बनाने में करती हूँ। तकनीक के साथ मेरा जुड़ाव सिर्फ प्रोफेशनल नहीं है, बल्कि यह मेरे लिए एक रचनात्मक यात्रा भी है। हर दिन कुछ नया सीखना और उसे अपने काम में लागू करना मुझे प्रेरित करता है।"

#         description = "Divya speaks in a warm, expressive tone with natural pauses and clear pronunciation. Her Hindi delivery is smooth and slightly animated, with moderate pitch and speed. The recording is very high quality, with her voice sounding intimate and close, capturing subtle emotional depth and realism."

#         print(f"\nInput Text (Hindi): {prompt}")
#         print(f"Description: {description}")

#         # ... (rest of the script is the same)
#         print("\nTokenizing inputs...")
#         input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
#         prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#         start_time = datetime.datetime.now()
#         print(f"Generating audio at Time Stamp {start_time}... This may take a moment.")
        
#         generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
#         audio_arr = generation.cpu().numpy().squeeze()
#         end_time = datetime.datetime.now()
#         print(f"Audio generated successfully.\n -----------------TAKES {end_time - start_time} MIN----------------- ")

#         output_filename = "Hindi_output_parler_local_correct.wav"
#         sampling_rate = model.config.sampling_rate
#         sf.write(output_filename, audio_arr, sampling_rate)
#         print(f"\nAudio successfully saved to '{os.path.abspath(output_filename)}'")

#     except Exception as e:
#         print(f"\nAn unexpected error occurred: {e}")

# if __name__ == "__main__":
#     generate_hindi_tts_from_local_structured()



import torch
from transformers import AutoTokenizer
import soundfile as sf
import os
import datetime
import numpy as np
import re # Import the regular expression library for better splitting

try:
    from parler_tts import ParlerTTSForConditionalGeneration
except ImportError:
    # ... (rest of the import check is the same)
    exit()

def generate_long_form_tts_robustly():
    """
    Generates long-form Hindi speech by splitting text into sentences, generating
    audio for each, and concatenating them with natural pauses.
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # --- Define paths (same as before) ---
        base_local_path = "./indic-parler-tts-local-structured"
        model_path = os.path.join(base_local_path, "model")
        prompt_tokenizer_path = os.path.join(base_local_path, "prompt_tokenizer")
        description_tokenizer_path = os.path.join(base_local_path, "description_tokenizer")

        print(f"Loading components from base path '{base_local_path}'...")
        if not os.path.exists(base_local_path):
            print("Error: Local model directory not found.")
            return

        # --- Load components (same as before) ---
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer_path)
        description_tokenizer = AutoTokenizer.from_pretrained(description_tokenizer_path)
        print("Model and tokenizers loaded successfully.")

        prompt = "नमस्ते! मेरा नाम अन्वी है। मैं एक तकनीकी उत्साही हूँ और मुझे पाइथन प्रोजेक्ट्स पर काम करना बेहद पसंद है। खासकर वे प्रोजेक्ट्स जो मशीन लर्निंग, डेटा एनालिटिक्स और कोड सुरक्षा से जुड़े होते हैं। जब भी मुझे कोई नया टूल या लाइब्रेरी मिलती है, तो मैं उसे एक्सप्लोर करने के लिए बहुत उत्साहित हो जाती हूँ। मुझे चुनौतियाँ पसंद हैं—चाहे वह मॉडल को ट्रेन करना हो, या किसी कोड को सुरक्षित बनाना। मैं अक्सर अपने समय का उपयोग नए कौशल सीखने, वर्कफ़्लो को बेहतर बनाने और अपने प्रोजेक्ट्स को अधिक प्रभावशाली बनाने में करती हूँ। तकनीक के साथ मेरा जुड़ाव सिर्फ प्रोफेशनल नहीं है, बल्कि यह मेरे लिए एक रचनात्मक यात्रा भी है। हर दिन कुछ नया सीखना और उसे अपने काम में लागू करना मुझे प्रेरित करता है।"
        description = "Divya speaks in a warm, expressive tone with natural pauses and clear pronunciation. Her Hindi delivery is smooth and slightly animated, with moderate pitch and speed. The recording is very high quality, with her voice sounding intimate and close, capturing subtle emotional depth and realism."

        print(f"\nFull Input Text (Hindi):\n{prompt}")

        # --- NEW LOGIC: Split the prompt into sentences using regex for both '।' and '.' ---
        sentences = re.split(r'(?<=[।\.])\s*', prompt)
        sentences = [s.strip() for s in sentences if s.strip()] # Remove empty strings
        print(f"\nSplitting text into {len(sentences)} sentences.")

        # --- NEW LOGIC: Define pause duration and create silence array ---
        sampling_rate = model.config.sampling_rate
        pause_duration_ms = 350  # A 350ms pause between sentences
        silence_samples = int(sampling_rate * (pause_duration_ms / 1000))
        silence_array = np.zeros(silence_samples, dtype=np.float32)

        all_audio_arrays = []
        start_time = datetime.datetime.now()
        print(f"Starting audio generation at {start_time}...")

        # --- NEW LOGIC: Loop through SENTENCES ---
        for i, sentence in enumerate(sentences):
            print(f"Generating audio for sentence {i+1}/{len(sentences)}: '{sentence}'")
            input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            all_audio_arrays.append(audio_arr)

            # Add the silence after each sentence, except the last one
            if i < len(sentences) - 1:
                all_audio_arrays.append(silence_array)

        # --- Concatenate all arrays (speech and silence) ---
        final_audio_arr = np.concatenate(all_audio_arrays)

        end_time = datetime.datetime.now()
        print(f"\nAll sentences generated and combined successfully.\n-----------------TOTAL TIME TAKEN: {end_time - start_time}-----------------")

        output_filename = "Hindi_output_long_form_FINAL.wav"
        sf.write(output_filename, final_audio_arr, sampling_rate)
        print(f"\nFull audio with natural pauses saved to '{os.path.abspath(output_filename)}'")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_long_form_tts_robustly()