import nltk
import unicodedata
from datasets import load_dataset, Dataset

nltk.download('punkt')


def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    return sentences


def remove_accents(text):
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in normalized if not unicodedata.combining(c)])


dataset = load_dataset('PleIAs/French-PD-Books', split='train', streaming=True)

all_sentences = []

nb_samples = 10000000
count = 0

for example in dataset:
    if count >= nb_samples:
        break
    sentences = split_into_sentences(example["complete_text"])
    for sentence in sentences:
        sentence_raw = remove_accents(sentence)
        all_sentences.append({
            "complete_text": sentence,
            "complete_text_raw": sentence_raw
        })
    count += 1

new_dataset = Dataset.from_list(all_sentences)
new_dataset.save_to_disk("./sentence_split_dataset")
print(new_dataset[0:5])
