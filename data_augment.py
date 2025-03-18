import os
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nltk
from googletrans import Translator
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load dataset
df = pd.read_csv("./Dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
df['instruction'] = df['instruction'].str.replace(r"[^\w\s]", "", regex=True)

# Set up directories
nltk_data_path = "./Dataset/nltk"
model_dir = "./models"
os.makedirs(nltk_data_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download NLTK data locally
nltk.download("wordnet", download_dir=nltk_data_path)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("omw-1.4", download_dir=nltk_data_path)

# Download and save the model locally
bert_model_path = os.path.join(model_dir, "bert-base-uncased")
if not os.path.exists(bert_model_path):
    print("Downloading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(bert_model_path)
    model.save_pretrained(bert_model_path)
    print("Model downloaded and saved to:", bert_model_path)
else:
    print("Model already exists in:", bert_model_path)

# Initialize augmenters with local models
syn_aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.3)  # Replace 30% words with synonyms
bert_aug = naw.ContextualWordEmbsAug(model_path=bert_model_path, action="substitute", aug_p=0.2)  # Use BERT for better augmentation

# Additional Augmentations for Precision
char_aug = nac.RandomCharAug(action="insert", aug_char_p=0.05)  # Minor typos for robustness
word_aug = naw.RandomWordAug(action="swap", aug_p=0.2)  # Swap words slightly for variation
sentence_aug = nas.ContextualWordEmbsForSentenceAug(model_path=bert_model_path)  # Generate variations of sentences

translator = Translator()  # Google Translate

# Function for back-translation
def back_translate(text, lang):
    try:
        translated = translator.translate(text, src="en", dest=lang).text
        back_translated = translator.translate(translated, src=lang, dest="en").text
        return back_translated
    except Exception as e:
        print(f"Error translating text: {e}")
        return text  # Return original if translation fails

# Function to apply all augmentations (Only modifies 'instruction')
def augment_text(text):
    augmented_texts = set()  # Use set to avoid duplicates
    
    if isinstance(text, str):  # Only check if text is a string
        augmented_texts.add(syn_aug.augment(text)[0])  # Synonym replacement
        augmented_texts.add(bert_aug.augment(text)[0])  # BERT contextual substitution
        augmented_texts.add(char_aug.augment(text)[0])  # Typo simulation
        augmented_texts.add(word_aug.augment(text)[0])  # Swap words for variation

        # Sentence-level augmentation (only for slightly longer texts)
        if len(text.split()) > 6:
            augmented_texts.add(sentence_aug.augment(text)[0])

        # Back-translation with multiple languages
        for lang in ["es", "fr"]:  # Spanish, French
            augmented_texts.add(back_translate(text, lang))

    return list(augmented_texts)

print("\n Augmentation functions loaded successfully!\n")

# Apply augmentation only on the "instruction" column
augmented_data = []
for _, row in df.iterrows():
    new_row = row.copy()  # Copy original row
    augmented_variants = augment_text(row["instruction"])  # Augment only "instruction"
    
    for variant in augmented_variants:
        new_row["instruction"] = variant  # Replace only "instruction"
        augmented_data.append(new_row.copy())  # Keep "category" and "intent" unchanged
        print(f"Appended: {variant}")

# Convert to DataFrame and save
aug_df = pd.DataFrame(augmented_data)
aug_df.to_csv("./Dataset/augmented_dataset.csv", index=False)

print(" Augmentation complete! Saved as 'augmented_dataset.csv'.")
