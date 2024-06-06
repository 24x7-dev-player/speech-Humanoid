import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')
cmu_dict = cmudict.dict()

def extract_phonemes(text):
    words = text.lower().split()
    phonemes = []
    for word in words:
        if word in cmu_dict:
            phonemes.extend(cmu_dict[word][0])
        else:
            phonemes.append(word)
    print(f"Phonemes: {phonemes}")
    return phonemes

# Sample usage
if __name__ == "__main__":
    text = "Hello, this is a test."
    phonemes = extract_phonemes(text)
