 # CogSolver & CogSolver+
Source code for ICDM-2022 paper *A Cognitive Solver with Autonomously Knowledge Learning for Reasoning Mathematical Answers* and manuscipt *Enhancing Mathematical Reasoning through Autonomously Learning Knowledge*.

 ## Dependencies
- python >= 3.6

- stanfordcorenlp
- torch == 1.8.1

 ## Usage
Train and test model
```bash
python main.py
```
For running arguments, please refer to [config.py](config.py).

The autonomouslly learned knowledge (as well as its strength) of CogSolver
* Math23K
```shell
knowledge\math23k\math23k_know_ww.txt: learned word-word relation knowledge
```
```shell
knowledge\math23k\math23k_know_wo.txt: learned word-operator relation knowledge
```
* MAWPS
```shell
knowledge\mawps\mawps_know_ww.txt: learned word-word relation knowledge
```
```shell
knowledge\mawps\mawps_know_wo.txt: learned word-operator relation knowledge
```

### Citation
If you find this work useful, please cite our paper:
```
@inproceedings{liu2022cognitive,
  title={A cognitive solver with autonomously knowledge learning for reasoning mathematical answers},
  author={Liu, Jiayu and Huang, Zhenya and Lin, Xin and Liu, Qi and Ma, Jianhui and Chen, Enhong},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  pages={269--278},
  year={2022},
  organization={IEEE}
}
```