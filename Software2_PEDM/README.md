# Software 2: protein egineering diffusion model

## Description
In this segment of our software, we have developed a sequence proposal model based on Denoising Diffusion Models in Discrete State-Spaces [1]. This model has been specifically tailored for the field of protein engineering, particularly when optimization objectives are guided by oracles.

Remarkably versatile, this model can accommodate a wide array of oracle functions, even those for which gradients cannot be explicitly derived. It is well-suited for situations where the ultimate objective possesses multiple facets, leading to the provision of multiple oracles for optimization.

In our specific application, we have employed this model to generate sequences that exhibit a remarkable 3.5-degree improvement over the original TcI within just seven iterations, with only 16 sequences during each iteration.

## Installation
The package dependencies for this software are listed in requirements.txt. To use this software, please run the following command.

```bash
git clone https://gitlab.igem.org/2023/software-tools/tsinghua.git
conda create -n PEDM python=3.8
conda activate PEDM
pip install -r requirements.txt
```

## Usage
The codes for this software are contained in '/src', and function arguments are regulated with hydra in 'cfg/config.yaml'
Before running the code, set the environment variable with the following code
```bash
cd tsinghua/Software2_PEDM/src
export ROOT=$(pwd)
```

### Training
Before sampling the sequence, train the model for an initial distribution close to the target sequence 
to reduce the variance involved in importance sampling:

First set the target sequence into seq in config.yaml. If this is the first time, set data.generate=True.
Then run the following code
```bash
python trainer.py task=train
```
### Sampling
Next, sample the sequence with the following code.
```bash
python trainer.py task=sample
```

### Iteration
Calculate the score for the optimization objectives with the oracle and set the corresponding path into iterate.temp_path in config.yaml
```bash
python trainer.py task=iterate
```

## Contributing
We are open to contributions. To contribute to this code base, please run the three task modes in trainer.py to ensure proper function of all parts.

## Authors and acknowledgment
Author: Mengchen Wang
Acknowledgment: We would like to extend our gratitude to the iGEM 2023 Tsinghua team for their invaluable support throughout the conception and implementation of this software. Special thanks go to Haochen Zhong, Yun Liu, Hongzhe Jia, Jiayi He, and Ruoying Wang for their engaging discussions and insights. These dialogues were instrumental in shaping the concept behind this software and in the process, eliminating several alternatives that proved challenging to implement or were flawed in conception.

Our appreciation also goes to our collaborative partners, the SJTU-software team, who contributed significantly to this endeavor. They developed a user-friendly and high-throughput monomer thermal stability prediction model, and their work on the web UI interface greatly facilitated the usability of the software.

## Reference
[1] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021. 