# FedPA

The implementation of **Aligning Foundation Models with Diverse User Preferences via Collaborative Pareto Optimization**. \
Pengxin Guo, Kunyang Song, Jinjing Zhu, Yuyin Zhou, Hui Xiong, and Liangqiong Qu.

<img src="./figs/FedPA.png" alt="framework" width="700" /> 

##### Figure 1. Overview of the FedPA framework. **Top:** The central server samples a preference vector $\boldsymbol{\alpha}$ and performs weighted aggregation of client updates. **Bottom:** Each client trains locally using the PCLoRA module, which injects the preference signal via a learned modulation matrix $\mathbf{W}(\boldsymbol{\alpha})$. Components $\mathbf{B}$, $\boldsymbol{\varphi}$, and $\mathbf{A}$ are updated via an alternating optimization strategy to avoid aggregation errors caused by naively averaging multiplicative low-rank parameters.


## Installation
Our code is based on [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft) for training and [vLLM](https://github.com/vllm-project/vllm) for inference. 
```
conda create -n fedpa python=3.10
conda activate fedpa

pip install vllm==v0.4.1 --extra-index-url https://download.pytorch.org/whl/cu121

cd ./peft/
pip install -e .

cd ..
git clone https://github.com/PKU-Alignment/safe-rlhf.git
cd safe-rlhf
pip install .

cd ..
pip install -r requirements.txt
```


## Training
```
cd code/training
bash run.sh
```

## Evaluation
```
cd code/evaluation
bash run.sh
```

