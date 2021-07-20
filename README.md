## Pufferfish
The implementation for MLSys 2021 paper: "Pufferfish: Communication-efficient Models At No Extra Cost"

### Overview
---
To mitigate communication overheads in distributed model training, several studies propose the use of compressed stochastic gradients, usually achieved by sparsification or quantization. Such techniques achieve high compression ratios, but in many cases incur either significant computational overheads or some accuracy loss. In this work, we present Pufferfish, a communication and computation efficient distributed training framework that incorporates the gradient compression into the model training process via training low-rank, pre-factorized deep networks. Pufferfish not only reduces communication, but also completely bypasses any computation overheads related to compression, and achieves the same accuracy as state-of-the-art, off-the-shelf deep models. Pufferfish can be directly integrated into current deep learning frameworks with minimum implementation modification. Our extensive experiments over real distributed setups, across a variety of large-scale machine learning tasks, indicate that Pufferfish achieves up to 1.64x end-to-end speedup over the latest distributed training API in PyTorch without accuracy loss. Compared to the Lottery Ticket Hypothesis models, Pufferfish leads to equally accurate, small-parameter models while avoiding the burden of "winning the lottery". Pufferfish also leads to more accurate and smaller models than SOTA structured model pruning methods.

### Depdendencies
---
Deep Learning AMI (Ubuntu 18.04) Version 40.0 (ami084f81625fbc98fa4) on Amazon EC2
(for FP32 results)
* PyTorch 1.4.0
* CUDA 10.1.243

(for AMP results)
* PyTorch 1.6.0
* CUDA 10.1.243

### Log files and model checkpoints for results reported
You can find the experiment log files and model checkpoints that are used for reporting the numbers in the paper
at [here](https://drive.google.com/drive/folders/18jfhpDfT80FK7YaZTGKQ6IlnxCNXsMZ-?usp=sharing)

### Sample commands to reproduce our experiments (single machine)
---
VGG-19 on CIFAR-10
```
python main.py \
--arch=vgg19 \
--mode=lowrank \
--batch-size=128 \
--epochs=300 \
--full-rank-warmup=True \
--fr-warmup-epoch=80 \
--seed=42 \
--lr=0.1 \
--resume=False \
--evaluate=False \
--ckpt_path=./checkpoint/vgg19_best.pth \
--momentum=0.9
```

ResNet-18 on CIFAR-10
```
python main.py \
--arch=resnet18 \
--mode=lowrank \
--batch-size=128 \
--epochs=300 \
--full-rank-warmup=True \
--fr-warmup-epoch=80 \
--seed=42 \
--lr=0.1 \
--resume=False \
--evaluate=False \
--ckpt_path=./checkpoint/vgg19_best.pth \
--momentum=0.9
```

ResNet-50/WideResNet-50 on ImageNet (ILSVRC2012)
```
python imagenet_training.py -a hybrid_resnet50 (hybrid_wide_resnet50_2) \
--vanilla-arch resnet50 (wide_resnet50_2) \
/path/to/imagenet/data/ \
--lr 0.1 \
--model-save-dir '/home/ubuntu/low-rank-ml' \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--mode=lowrank \
--full-rank-warmup=True \
--re-warmup=True \
--fr-warmup-epoch=15 \
--lr-warmup= \
--warmup-epoch=5 \
-j 8 \
-p 40 \
--multiplier=16 \
-b 256
```

LSTM on WikiText-2
source code at: `Pufferfish/word_language_model`
```
python main.py \
--cuda \
--lr 20 \
--emsize 1500 \
--nhid 1500 \
--dropout 0.65 \
--epochs 40 \
--warmup True \
--warmup-epoch 10 \
--tied
```

Transformer on WMT16'
source code at: `Pufferfish/low_rank_transformer`
```
python train.py -data_pkl m30k_deen_shr.pkl \
-log m30k_deen_shr \
-embs_share_weight \
-proj_share_weight \
-label_smoothing \
-save_model best \
-b 256 \
-warmup 128000 \
-epoch 400 \
-seed 0 \
-fr_warmup_epoch 10
```

### Sample commands to reproduce our experiments (distributed)
---
The source code to reproduce our distributed experiments is in `Pufferfish/dist_experiments`
The following sample commands are all for distributed CIFAR-10 experiments on ResNet-18

Vanilla SGD:
```
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor vanilla \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.44.194:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}
```

SIGNUM:
```
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor signum \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.2.165:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}
```

Pufferfish:
(where Powerfish means we run PowerSGD for full-rank warmup epochs in Pufferfish)
```
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor powersgd \
--config-mode powerfish \
--distributed \
--master-ip 'tcp://172.31.0.8:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}
```

Pufferfish+PowerSGD:
```
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor powersgd \
--config-mode pufferfish \
--distributed \
--master-ip 'tcp://172.31.0.8:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}
```

PowerSGD:
```
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor powersgd \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.6.30:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}
```

### Citing Pufferfish
---
If you found the code/scripts here are useful to your work, please cite Pufferfish by
```
@article{wang2021pufferfish,
  title={Pufferfish: Communication-efficient Models At No Extra Cost},
  author={Wang, Hongyi and Agarwal, Saurabh and Papailiopoulos, Dimitris},
  journal={Proceedings of Machine Learning and Systems},
  volume={3},
  year={2021}
}
```
