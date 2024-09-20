import nltk
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_punkt_tab():
    nltk.download('punkt')
    punkt_path = nltk.data.find('tokenizers/punkt').path
    tab_file = os.path.join(punkt_path, 'english.pickle')

    if not os.path.exists(tab_file):
        print(f"Creating {tab_file}")
        with open(tab_file, 'w') as f:
            f.write('')
    else:
        print(f"{tab_file} already exists")

if __name__ == "__main__":
    download_punkt_tab()
    nltk.download('averaged_perceptron_tagger')
    print("NLTK setup complete.")