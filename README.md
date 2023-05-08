# GraphFit: Learning Multi-scale Graph-convolutional Representation for  Point Cloud Normal Estimation
PyTorch implementation of paper "GraphFit: Learning Multi-scale Graph-convolutional Representation 
for Point Cloud Normal Estimation", ECCV 2022.

## Installation
Clone this repo:
```
git clone https://github.com/UestcJay/GraphFit.git
cd GraphFit/
```
The code is tested with Ubuntu16.04, Python3.7, PyTorch == 1.6.0 and CUDA == 10.2. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. we conduct the experiment in the following setting:
```
pytorch==1.6.0
torchvision==0.7.0
numpy==1.19.2
matplotlib==3.3.4
scikit-learn==0.21.3
scipy==1.6.0
urllib3==1.26.3
tensorboardX==2.2
```
## Datasets
```
├──data/
    ├──pcpnet/
```
Run `get_data.py` to download PCPNet data.
Alternatively, Download the PCPNet data from this [link](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) and place it in  `./data/pcpnet/` directory.

## Training
when `k=256, batch_size=256`, we use 2 `Tesla V100`.
```
python train_n_est.py
```

## Evaluation
```
# To test the model and output all normal estimations for the dataset run
python test_n_est.py
# To evaluate the results and output a report 
python evaluate.py
```
If you would like to use the given model, you can ref the [issue](https://github.com/UestcJay/GraphFit/issues/3#issuecomment-1537456649).
## Acknowledgement
The code is heavily based on [DeepFit](https://github.com/sitzikbs/DeepFit).

If you find our work useful in your research, please cite the following papers.

```
@inproceedings{li2022graphfit,
  title={GraphFit: Learning Multi-scale Graph-convolutional Representation 
for Point Cloud Normal Estimation},
  author={Keqiang Li, Mingyang Zhao, Huaiyu Wu, Dong-Ming Yan, Zhen Shen, Fei-Yue Wang and Gang Xiong},
  booktitle={European conference on computer vision},
  year={2022},
  organization={Springer}
}

@inproceedings{zhu2021adafit,
  title={AdaFit: Rethinking Learning-based Normal Estimation on Point Clouds},
  author={Zhu, Runsong and Liu, Yuan and Dong, Zhen and Wang, Yuan and Jiang, Tengping and Wang, Wenping and Yang, Bisheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6118--6127},
  year={2021}
}

@inproceedings{ben2020deepfit,
  title={Deepfit: 3d surface fitting via neural network weighted least squares},
  author={Ben-Shabat, Yizhak and Gould, Stephen},
  booktitle={European conference on computer vision},
  pages={20--34},
  year={2020},
  organization={Springer}
}
```
