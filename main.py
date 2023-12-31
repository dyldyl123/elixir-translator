import pytesseract as pt
import numpy as np
from PIL import Image
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# print(data_list)
list_of_items = []
for item in data_list:
    descriptions = item.get("descriptions")
    if isinstance(descriptions,list):
        for description in descriptions:
            list_of_items.append(description)
pt.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Replace with the path from the 'which tesseract' command
img_path = '/Users/dylanfitzmaurice/Desktop/left.png'
image = cv2.imread(img_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Define lower and upper thresholds for yellow color
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)

# Create a binary mask for yellow color
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply the mask to the original image
yellow_text = cv2.bitwise_and(image, image, mask=yellow_mask)

# Convert the yellow text region to grayscale
gray_yellow_text = cv2.cvtColor(yellow_text, cv2.COLOR_BGR2GRAY)

# Use PyTesseract to do OCR on the yellow text region
text = pt.image_to_string(Image.fromarray(gray_yellow_text))

# Print the extracted yellow text
print("Yellow Text:")
print(text)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over contours and filter based on area (you may need to adjust this threshold)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

# Create a mask for the filtered contours
contour_mask = cv2.drawContours(np.zeros_like(binary_image), filtered_contours, -1, (255), thickness=cv2.FILLED)

# Apply the mask to the original image
white_text = cv2.bitwise_and(image, image, mask=contour_mask)

# Convert the white text region to grayscale
gray_white_text = cv2.cvtColor(white_text, cv2.COLOR_BGR2GRAY)

# Use PyTesseract to do OCR on the white text region
text_white = pt.image_to_string(Image.fromarray(gray_white_text))

# Print the extracted white text
print("White Text:")
text_white = text_white.replace("Iwill", "I will")
print(text_white)

text_file_path = 'translations.txt'
read_data_tuples = []
korean_phrases_translated = []
with open(text_file_path, 'r', encoding='utf-8') as text_file:
    for line in text_file:
        original_text, translated_text = line.strip().split('\t')
        read_data_tuples.append((original_text, translated_text))

# Print the read data
for original, translated in read_data_tuples:
    korean_phrases_translated.append(translated)
    print(f"Original Text: {original}")
    print(f"Translated Text (English): {translated}\n")


english_phrases = [text_white]



# # Sample Korean phrases translated to English
# korean_phrases_translated = ["안녕하세요, 어떻게 지내세요?", "당신의 이름은 무엇인가요?", "안녕!"]

# Vectorize English and translated phrases using TF-IDF
vectorizer = TfidfVectorizer()
english_vectors = vectorizer.fit_transform(english_phrases)
translated_vectors = vectorizer.transform(korean_phrases_translated)

# Calculate cosine similarity between English and translated vectors
similarity_matrix = cosine_similarity(english_vectors, translated_vectors)

# Find the best match for each English phrase
matches = [korean_phrases_translated[similarity.argmax()] for similarity in similarity_matrix]

# Print the matches
for english, match in zip(english_phrases, matches):
    print(f"English: {english}")
    print(f"Match (Korean): {match}\n")
    korean_result = [item[0] for item in read_data_tuples if match in item]
    print(f"korean result: {korean_result}")
    
