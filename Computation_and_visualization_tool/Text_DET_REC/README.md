# Text Detection and Recognition Module

This module performs text detection and recognition on chart Image

we adapted CRAFT text detection technique for isolating textual regions from image.
| [Paper](https://arxiv.org/abs/1904.01941) | [Code](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [Pretrained model](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) |



we adapted four-stage STR framework designed along with CRAFT model for recognizing the text regions recognized.
| [Paper](https://arxiv.org/abs/1904.01906) | [Code](https://github.com/clovaai/deep-text-recognition-benchmark) | [Pretrained model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) | 
 We also use Tessract text recognition module along side in few cases for better accuracy.

```
pip install pytesseract
```
Things to be taken care before runing the code:
1. Reset the path of pretrained model in Text_DET_REC/CRAFT_TextDetector/__Init__.py, line 47, 160, and 176
2. Reset the path of pretrained model in Text_DET_REC/Deep_TextRecognition/__Init__.py, line 95, and 98

### To run the module
1. The [Retrieve_Text.py](url) is used to retrieve text from the specified locations of the chart image with it's respective funtion call
2. The funtion calls takes both image file and it's annotated xml file as input
3. These function calls are used in [Chart_Reconstruct](url) module
4. The [Draw_text_boxes.py](url) is used to visalize text recognition and detection results of chart image using both Tesseract API and above discussed method


