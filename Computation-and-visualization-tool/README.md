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
