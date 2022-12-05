# HAKG

<div align=center>
<img src=./fig/framework.png width="80%" ></img>
</div>

This is our Pytorch implementation for the paper:

> Yuntao Du, Xinjun Zhu, Lu Chen, Baihua Zheng and Yunjun Gao. (2022). HAKG: Hierarchy-Aware Knowledge Gated Network for Recommendation. Paper in [ACM DL](https://dl.acm.org/doi/10.1145/3477495.3531987) or Paper in [arXiv](https://arxiv.org/abs/2204.04959). In SIGIR'22, Madrid, Spain, July 11–15, 2022.

## Introduction

Hierarchy-Aware Knowledge Gated Network (HAKG) is a new recommendation framework tailored to knowledge-aware recommendation. Built upon the hyperbolic space and graph neural network framework, HAKG explicitly models the hierarchical structures and relations in user-item graph and KG, and propose a novel dual item embeddings design to represent and propagate collaborative signals and knowledge associations separately.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{HAKG22,
  author    = {Yuntao Du and
               Xinjun Zhu and
               Lu Chen and
               Baihua Zheng and 
               Yunjun Gao},
  title     = {{HAKG:} Hierarchy-Aware Knowledge Gated Network for Recommendation},
  pages     = {1390--1400},
  booktitle = {{SIGIR}},
  year      = {2022}
}
```

## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)
- PyTorch == 1.8
- networkx == 2.5
- A Nvidia GPU with cuda 11.1+

## Datasets

We user three popular datasets: Alibaba-iFashion, Yelp2018 and Last-FM to conduct experiments.
* We follow the paper "[Learning Intents behind Interactions with Knowledge Graph for Recommendation]()" to process data.
* You can find the full version of recommendation datasets via [Alibaba-iFashion](https://github.com/wenyuer/POG), [Yelp2018](https://www.heywhale.com/mw/dataset/5ecbc342fac16e0036ec41a0) and [Last-FM](http://www.cp.jku.at/datasets/LFM-1b/).
* [New] We have added all datasets in our paper, and it can be found in [here](https://github.com/zealscott/HAKG/tree/main/data).

## Reproducibility & Training

To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts, and provide [the log for our trainings](./result/).

- Alibaba-iFashion dataset
```shell
python main.py --dataset alibaba-ifashion --lr 0.0001 --angle_loss_w 0.005 --context_hops 3 --num_neg_sample 200 --margin 0.6
```

- Yelp2018 dataset
```shell
python main.py --dataset yelp2018 --lr 0.0005 --angle_loss_w 0.005 --context_hops 2 --num_neg_sample 400 --margin 0.8
```

- Last-FM dataset
```shell
python main.py --dataset last-fm --lr 0.0001 --angle_loss_w 0.005 --context_hops 3 --num_neg_sample 400 --margin 0.7
```

## Acknowledgement
Any scientific publications that use our datasets should cite the following paper as the reference:

```
@inproceedings{HAKG22,
  author    = {Yuntao Du and
               Xinjun Zhu and
               Lu Chen and
               Baihua Zheng and 
               Yunjun Gao},
  title     = {{HAKG:} Hierarchy-Aware Knowledge Gated Network for Recommendation},
  booktitle = {{SIGIR}},
  year      = {2022}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.
