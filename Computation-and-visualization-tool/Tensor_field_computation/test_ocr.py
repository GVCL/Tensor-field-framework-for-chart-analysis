from PIL import Image
import pytesseract

im = Image.open("bar13.png")

text = pytesseract.image_to_string(im, lang='eng')

print(text)