python main_lth.py \
--arch=vgg19 \
--mode=lowrank \
--batch-size=128 \
--epochs=160 \
--full-rank-warmup=True \
--fr-warmup-epoch=50 \
--seed=1 \
--lr=0.1 \
--momentum=0.9 > log 2>&1


# python main_lth.py \
# --arch=vgg19 \
# --mode=lowrank \
# --batch-size=128 \
# --epochs=160 \
# --full-rank-warmup=True \
# --fr-warmup-epoch=75 \
# --evaluate=True \
# --resume=True \
# --ckpt_path=./checkpoint/vgg19_best.pth \
# --lr=0.1 \
# --momentum=0.9