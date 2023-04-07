# Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. SIGIR22
## Example to Run the Codes
This is our Tensorflow implementation for the paper. Using the following command to run this code (see the parser function in src/utility/parser.py to get the meaning of more parameters):
> python3 main.py --dataset amazon-book --alg_type lightgcn --adj_type norm --lr 0.001 --batch_size 1024 --regs [1e-4] --layer_size [64,64,64] --alpha 1
## Environment Requirement
- tensorflow == 1.15.0
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
