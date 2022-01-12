# SuperAdam
Official Pytorch implementation for the paper 'SUPER-ADAM: Faster and Universal Framework of Adaptive Gradients' (https://openreview.net/pdf?id=nFdJSm9dy83)

This repository contains pytorch implementation for the SuperAdam optimizer and an example using SuperAdam to train  ResNet-18 over CIFAR-10 dataset


## Installation

To use the code, run:

```
git clone https://github.com/LIJUNYI95/SuperAdam
```

The current version of the repo uses the following packages: numpy and pytorch.
If you do not have these packages, please run the following command to install them:
 
```
 pip install -r requirements.txt
```

## Using our code
Run

 ```
python3 main.py
```
This uses SuperAdam to train a ResNet-18 network over CIFAR-10 dataset.



## Citation

If you publish material that uses this code, you can use the following citation:

```js
@article{huang2021super,
  title={SUPER-ADAM: Faster and Universal Framework of Adaptive Gradients},
  author={Huang, Feihu and Li, Junyi and Huang, Heng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```



