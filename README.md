# GeniePath-pytorch
This is a PyTorch implementation of the GeniePath model in [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)

> GeniePath, a scalable approach for learning adap- tive receptive fields of neural networks defined on permuta- tion invariant graph data. In GeniePath, we propose an adap- tive path layer consists of two complementary functions de- signed for breadth and depth exploration respectively, where the former learns the importance of different sized neighbor- hoods, while the latter extracts and filters signals aggregated from neighbors of different hops away. Our method works in both transductive and inductive settings, and extensive ex- periments compared with competitive methods show that our approaches yield state-of-the-art results on large graphs


## Model

<img src= "https://github.com/shawnwang-tech/GeniePath-pytorch/blob/master/doc/GeniePath.png"/>

<img src= "https://github.com/shawnwang-tech/GeniePath-pytorch/blob/master/doc/GeniePathLazy.png"/>


## Requirements

- [pyG](https://github.com/rusty1s/pytorch_geometric)


## Usage

### Install packages

```bash
pip install -r requirements.txt
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Choose Model

in ppi_geniepath.py

```bash
# from model_geniepath import GeniePath as Net
from model_geniepath import GeniePathLazy as Net
```

### Run

```bash
python ppi_geniepath.py
```


## TODO

- [x] Finish the rough implementation, f1_score: 0.9709
 for GeniePath,  0.9762 for GeniePathLazy (dim = 256, lstm_hidden = 256). 
- [ ] Tune the model


## Reference
- [pytorch_geometric/examples/ppi.py](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ppi.py)