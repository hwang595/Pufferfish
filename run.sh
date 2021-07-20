python main.py \
--arch=vgg19 \
--mode=lowrank \
--batch-size=128 \
--epochs=300 \
--full-rank-warmup=True \
--fr-warmup-epoch=30 \
--seed=42 \
--lr=0.1 \
--resume=False \
--evaluate=False \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9 > log 2>&1



# python main.py \
# --arch=vgg19 \
# --mode=vanilla \
# --batch-size=128 \
# --epochs=300 \
# --full-rank-warmup=False \
# --fr-warmup-epoch=30 \
# --seed=42 \
# --lr=0.1 \
# --resume=False \
# --evaluate=False \
# --ckpt_path=./checkpoint/resnet18_best.pth \
# --momentum=0.9 > log 2>&1

# python main.py \
# --arch=resnet18 \
# --mode=lowrank \
# --batch-size=128 \
# --epochs=300 \
# --full-rank-warmup=True \
# --fr-warmup-epoch=80 \
# --seed=42 \
# --evaluate=True \
# --resume=True \
# --ckpt_path=./checkpoint/resnet18_best.pth \
# --lr=0.1 \
# --momentum=0.9


for SEED in 1 2 3
do
  python main.py \
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
  --momentum=0.9 > pufferfish_resnet18_cifar10_seed${SEED}_log 2>&1
done