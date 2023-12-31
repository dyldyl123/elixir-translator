import pytesseract as pt
import numpy as np
from translate import Translator
from dataJson import data_list
# print(data_list)
list_of_items = []
for item in data_list:
    descriptions = item.get("descriptions")
    if isinstance(descriptions,list):
        for description in descriptions:
            list_of_items.append(description)

output = []
print(len(list_of_items))
for idx,item in enumerate(list_of_items):
    try:
        translator_to_english = Translator(from_lang = "ko",to_lang = "en")
        translated_individual_text = translator_to_english.translate(item)
        print(idx)
        output.append(translated_individual_text)
    except Exception as e:
        print(f'Error translating: {e}')
        output.append(None)


for original, translated in zip(list_of_items, output):
    print(f"Original Text: {original}")
    print(f"Translated Text (Korean): {translated}\n")


# Combine original and translated lists into tuples
data_tuples = list(zip(list_of_items, output))

# Write tuples to a text file
text_file_path = 'translations.txt'
with open(text_file_path, 'w', encoding='utf-8') as text_file:
    for data_tuple in data_tuples:
        text_file.write('\t'.join(data_tuple) + '\n')

print(f"Data written to {text_file_path}")
