import pytesseract
from PIL import Image

# Важно для Windows: указать путь к tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Открываем изображение с помощью Pillow
img = Image.open('number_train/orig.jpg')

# Используем image_to_string для распознавания текста
text = pytesseract.image_to_string(img, lang='rus+eng') # lang указывает язык(и)

print(text)