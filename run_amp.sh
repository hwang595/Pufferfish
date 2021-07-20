for SEED in 1 2 3
do
  python main_amp.py \
  --arch=resnet18 \
  --mode=lowrank \
  --batch-size=128 \
  --epochs=300 \
  --full-rank-warmup=True \
  --fr-warmup-epoch=80 \
  --seed=${SEED} \
  --lr=0.1 \
  --resume=False \
  --evaluate=False \
  --ckpt_path=./checkpoint/resnet18_best.pth \
  --momentum=0.9 > pufferfish_resnet18_amp_cifar10_seed${SEED}_log 2>&1
done