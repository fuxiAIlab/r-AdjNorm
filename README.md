# Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. SIGIR22
## Introduction
This is our Tensorflow implementation for the paper. Our codes are adapted from the following repo:
> https://github.com/xiangwang1223/neural_graph_collaborative_filtering. 

On this basis, we have implemented the following baselines:
> PPNW: Kachun Lo and Tsukasa Ishigaki. Matching Novelty While Training: Novel Recommendation Based on Personalized Pairwise Loss Weighting. In ICDM 2019. 468–477.

> Reg&PC:Ziwei Zhu, Yun He, Xing Zhao, Yin Zhang, Jianling Wang, and James Caverlee. Popularity-Opportunity Bias in Collaborative Filtering. In WSDM 2021. 85–93.

> DegDrop (DropEdge with degree perference): Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. In ICLR 2019.

## Usages
Using the following command to run this code (see the parser function in src/utility/parser.py to get the meaning of more parameters):
> python3 main.py --dataset amazon-book --alg_type lightgcn --adj_type norm --lr 0.001 --batch_size 1024 --regs [1e-4] --layer_size [64,64,64] --r 1
## Requirements
- tensorflow == 1.15.0
- numpy == 1.16.4
- scipy == 1.2.1
- scikit-learn == 0.20.2
## Reference
If you use these codes, please cite the following paper, thank you:
```
@inproceedings{zhao2022investigating,
  title={Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering},
  author={Zhao, Minghao and Wu, Le and Liang, Yile and Chen, Lei and Zhang, Jian and Deng, Qilin and Wang, Kai and Shen, Xudong and Lv, Tangjie and Wu, Runze},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={50--59},
  year={2022}
}
```
