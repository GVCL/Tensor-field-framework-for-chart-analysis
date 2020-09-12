# Computation-and-visualization-tool
Seperate modules for tensor field computations and visualizations

For Tensor field computation and generate csv files:
```
1. cd Tensor_field_computation
2. python image_uploader.py
3. upload required image
```

This wil generate the folowing csv files:
1. Image_RGB.csv: contains RGB values along with xy-coordinates.
2. structure_tensor.csv: contains structure tensor matrix, eigen values, eigen vectors and cl-cp values.
3. tensor_vote_matrix.csv: contains tensor voting matrix, eigen values, eigen vectors and cl-cp values.


For Tensor field visualization:
```
1. cd Visualizer
2. python csv_uploader.py
3. upload required csv file for structure tensor/tensor vote visualization.
```

Future Work:
- [ ] Add interactivity to toggle between hedgehog and scatter plots.
- [ ] Add stramline plots.

## Text Detection and Recognition Module

This module performs text detection and recognition on chart Image

we adapted CRAFT text detection technique for isolating textual regions from image.
| [Paper](https://arxiv.org/abs/1904.01941) | [Code](https://github.com/clovaai/CRAFT-pytorch) |

we adapted four-stage STR framework designed along with CRAFT model for recognizing the text regions recognized.
| [Paper](https://arxiv.org/abs/1904.01906) | [Code](https://github.com/clovaai/deep-text-recognition-benchmark) |
 We also use Tessract text recognition module along side in few cases for better accuracy.

```
pip install pytesseract
```
or
```
install tesseract along
brew install tesseract
```
This formula contains only the "eng", "osd", and "snum" language data files.
If you need any other supported languages, run `brew install tesseract-lang`.

Things to be taken care before runing the code:
1. Download the pretrained model [craft_mlt_25k.pth](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view), then reset the path in Text_DET_REC/CRAFT_TextDetector/__Init__.py, line 47.
2. Download the pretrained model [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW), then reset the path of pretrained model in Text_DET_REC/Deep_TextRecognition/__Init__.py, line 98.

#### To run the module
1. The [Retrieve_Text.py](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Computation_and_visualization_tool/Text_DET_REC/Retrieve_Text.py) is used to retrieve text from the specified locations of the chart image with it's respective funtion call
2. The [Draw_text_boxes.py](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Computation_and_visualization_tool/Text_DET_REC/Draw_text_boxes.py) is used to visalize text recognition and detection results of chart image using both Tesseract API and above discussed method

## Chart Reconstruction Module

This module performs Chart Reconstruction. It is handled by [data_extract.py](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Computation_and_visualization_tool/Tensor_field_computation/data_extract.py) based on chart classifier type
