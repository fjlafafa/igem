# Software 1: PPM
PPM, Promoter Promotion Model, is a software tool that can predict the strength of prokaryotes promoters. On the same dataset, the performance of PPM outperforms [the published model][1] on Nucleic Acids Research by **50%**. In addition, we further optimized the results of the Tsinghua iGEM 2022 team, reducing the model memory footprint to **1/20** of the original, improving the prediction effect by **22%**, and optimizing the user interface and usage instructions. This makes our model easier to deploy and apply.

## Description
Promoter optimization is a fundamental but important component in the field of synthetic biology. In the past, researchers needed to perform complex directed evolution to screen, which consumed a lot of time and effort. With the rise of machine learning and deep learning, we can now use language models to accurately predict promoter strength. Here we propose two new models to solve the problem of promoter strength prediction. The first model uses convolutional neural networks, LSTM, and attention mechanisms; The second model uses Densenet and CBAM attention mechanisms. Compared to existing models, PPM achieves better results and consumes less memory. We have also written detailed documentation to facilitate user use and deployment.

## Installation
The environment needs:
- python>=3.8
- pytorch>=1.11
- lightning>=2.0.0
- torchmetrics>=0.7.0
- pandas
  
You can install the environment by running the following command:
```bash
conda create -n ppm python=3.8
conda activate ppm
conda install pytorch pandas lightning torchmetrics -c pytorch -c conda-forge
```
You can also install the environment by running the following command:
```bash
pip install -r requirements.txt
```

## Usage
### Random Mutation(Optional)
If you already have the promoter sequence you need to predict, you can skip this step.

If you don't have the promoter sequence you need to predict, you can use the following command to generate random promoter sequences:

```bash
cd PPM
python script/mutation.py --mode <mode> -i <input_file> -o <output_file> \
--num <number of loops in mode 1> --pos <position of mutation in mode 2>
```
- `--mode`: 1 or 2, 1 means random mutation in whole sequence, 2 means mutation at the specific position
- `-i/--input`: the path of the input file. The input file should contain the promoter sequence with a length of 50bp.
- `-o`: the path of the output file. The output file will contain the mutated promoter sequence.
- `--num`: the number of loops in mode 1.
- `--pos`: the position of mutation in mode 2.

For example:
```bash
python ./script/mutation.py -i ./example/PnisA.txt -o ./example/mutation.txt --mode 1 --num 2000
python ./script/mutation.py -i ./example\PnisA.txt -o ./example/mutation.txt --mode 2 --pos 1,5
```

### Prediction
We provide two models for prediction, you can choose one of them to use. Model 1 is a model based on CNN, LSTM, and attention mechanism. Model 2 is a model based on Densenet and CBAM attention mechanism. The two models have similar prediction effects. Model 1 has a smaller memory footprint and faster prediction speed while model 2 has a higher average prediction effect. You can choose the model according to your needs.

You can use the following command to predict the strength of the promoter sequence:

```bash
cd PPM
python script/main.py --mode <mode> -i <input_file> -o <output_dir_path> --model <model_path> --batch_size <batch_size> --sort
```

- `--mode`: 1 or 2, 1 means model 1, 2 means model 2.
- `-i/--input`: the path of the input file. The input file should contain the promoter sequence with a length of 50bp.
- `-o`: the path of the output directory. The output directory will contain the prediction result.
- `--model`: the path of the model.
- `--batch_size`: the batch size of the prediction. The default value is 64.
- `--sort`: whether to sort the prediction result. If you want to sort the prediction result, you can add this parameter.

For example:
```bash
python ./script/main.py -i ./example/promoter.csv -o ./example/ --model ./model/best_model_1.ckpt --mode 1 --batch_size 64 --sort
```

## Contributing
We are open to contributions.

## Authors and acknowledgment
**Author:** Haochen Zhong
**Github:** [hurther](https://github.com/hurther)
**Acknowledgment:** 
- Thanks to the Tsinghua iGEM 2022 team for providing the model training data!
- Thanks to Zihan Qiu for providing GPU computing power support for the model training and detailed guidance!!
- Thanks to the Tsinghua iGEM 2023 team for their continuous support and companionship!!!

## References
[1]: Ye Wang and others, Synthetic promoter design in Escherichia coli based on a deep generative network, Nucleic Acids Research, Volume 48, Issue 12, 09 July 2020, Pages 6403–6412, https://doi.org/10.1093/nar/gkaa325