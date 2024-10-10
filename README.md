# FastAdaptiveLLIE
Gamma boost for low-light image enhancement with an adaptive selector network

# Structure
```
├─📄 assigned_data.py
├─📁 docs-------------------------------------- # results on PIQA eval dataset
│ ├─📄 box_plot.png
│ ├─📄 heatmap.png
│ └─📄 scatter_plot.png
├─📄 evalGraph.py------------------------------ # draw graphs for evaluation
├─📄 evalMethod.py----------------------------- # training analysis
├─📄 evalPIQAtest.py
├─📄 ForwardNN.py------------------------------ # training
├─📁 HE_Based
│ ├─📄 illumination_boost.py
│ └─📁 __pycache__
│   └─📄 illumination_boost.cpython-311.pyc
├─📁 input_images------------------------------ # test images for assigned_data.py
├─📄 mapping_nn_model1.pth
└─📁 PIQA-dataset-main
  └─📁 PIQA-dataset-main
    ├─📁 checkpoints
    │ ├─📄 info.txt
    │ └─📄 PIQA.keras
    ├─📄 converted.py
    ├─📁 dataset
    ├─📄 README.md
    └─📄 test.py
```
# Reference
```
@article{rasheed2022comprehensive,
  title={A comprehensive experiment-based review of low-light image enhancement methods and benchmarking low-light image quality assessment},
  author={Rasheed, Muhammad Tahir and Shi, Daming and Khan, Hufsa},
  journal={Signal Processing},
  pages={108821},
  year={2022},
  publisher={Elsevier}
}
```
