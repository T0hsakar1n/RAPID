# RAPID

This official repository contains the source code and scripts for the paper "*[Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks](https://arxiv.org/abs/2409.00426)*," accepted by [ACM CCS 2024](https://www.sigsac.org/ccs/CCS2024/home.html), authored by Yu He, Boheng Li, and Yao Wang et al. 

In this paper, we propose RAPID (**R**e-lever**A**ging original membershi**P** scores to m**I**tigate errors in **D**ifficulty calibration), a novel framework for Membership Inference Attacks (MIAs). It provides a practical and effective approach to better understand and mitigate the risks posed by MIAs, achieving significant advancements in precision and computational efficiency.

## Content
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Online RAPID Attack](#online-rapid-attack)

## Getting Started

### Setup

Our code has been tested on Linux (a server with NVIDIA A6000 GPUs, each with 48GB memory) with Python 3.9.20, CUDA 12.1, PyTorch 2.0.1

To set up the environment, follow these three steps:

1. Clone this repository 
```bash
git clone https://github.com/T0hsakar1n/Is-Difficulty-Calibration-All-We-Need-Towards-More-Practical-Membership-Inference-Attacks.git
cd Is-Difficulty-Calibration-All-We-Need-Towards-More-Practical-Membership-Inference-Attacks
```

2. Install CUDA 12.1, pytorch 2.0.1, python 3.9 within a `conda` virtual environment.
```bash
conda create -n rapid python=3.9
conda activate rapid
pip install numpy==1.23.0 torch==2.0.1
```
3. Run the following command to install the other required packages listed in the `requirements.txt` file in the current directory:
```bash
pip install -r requirements.txt
```

4. Run the following Python script to check if the GPU and CUDA environment are correctly recognized and available for use:

   ```python
   import torch
   
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```

   If `torch.cuda.is_available()` returns `True`, the environment is ready. 

### Data
- The source code will automatically download the required datasets in the subsequent steps, so there is no need to download them separately. For details on how the datasets are retrieved, please refer to the **datasets.py** file.

### Training

Here, we use VGG16 model and CIFAR10 dataset as an example to reproduce the main results from the paper:

1. Train the victim model and shadow model
```bash
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
```

2. Train the reference models
```bash
python refer_model.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 8
```
- Optionally, use distributed training (not recommend) 
```bash
python refer_model.py config/cifar10/cifar10_vgg16.json --distributed True --world_size 4 --model_num 8
```

### Evaluation
1. Once you have trained the models, you can evaluate the attack's effectiveness using the following commands:
```bash
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json  --attack_rapid
```
2. Use the following commands to generate the corresponding ROC curve images:
```bash
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack
```

### Online RAPID Attack

The following commands demonstrate how to perform the RAPID attack in an online setting (assuming you have already trained the victim model and shadow model) :
```bash
python refer_model_online.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 256 --state victim
python refer_model_online.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 256 --state shadow
python mia_attack_online.py 0 config/cifar10/cifar10_vgg16.json --model_num 256 --attack_rapid
```

## Contact the developers

If you've found a bug or are having trouble getting code to work, please feel free to open an issue on the [<u>GitHub repo</u>](https://github.com/T0hsakar1n/Is-Difficulty-Calibration-All-We-Need-Towards-More-Practical-Membership-Inference-Attacks). 

## Citation

If you find our work helpful in your research, please cite it using the following:
```bibtex
@inproceedings{he2024difficulty,
      title={Is Difficulty Calibration All We Need? Towards More Practical Membership Inference Attacks}, 
      author={He, Yu and Li, Boheng and Wang, Yao and Yang, Mengda and Wang, Juan and Hu, Hongxin and Zhao, Xingyu},
      booktitle={CCS},
      year={2024},
}
```