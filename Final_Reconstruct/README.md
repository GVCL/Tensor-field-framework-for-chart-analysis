# Chart Reconstruction Module

This is the main module that performs Chart Reconstruction

Things to be taken care before runing the code:
1. We run the [preprocess_gentensorvote.py](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Final_Reconstruct/preprocess_gentensorvote.py) to preprocess the image for tensor voting computation. The [segment](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Chart_Seg/Graph_Obj_Seg.py) funtion is called to segment the graphical objects from image, we then add borders to objects.
2. Now the [compute_tensorvote.py](https://github.com/GVCL/Tensor-field-framework-for-chart-analysis/blob/master/Final_Reconstruct/compute_tensorvote.py) is used compte tensor vote for preprocessed image.

### To run the module
1. Set chart-type variable and path variable to location of directory containing Image file and XML file in Final_Reconstruct/preprocess_gentensorvote.py, line 7, and 8
2. Now run the respective reconstruction file based on chart-type by setting path variable
