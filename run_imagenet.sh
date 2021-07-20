python imagenet_training.py -a hybrid_resnet50 \
--vanilla-arch resnet50 \
--dist-url 'tcp://127.0.0.1:1234' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
~/data/ \
--lr 0.1 \
--model-save-dir '/home/ubuntu/low-rank-ml' \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--mode=lowrank \
--full-rank-warmup=True \
--re-warmup=True \
--fr-warmup-epoch=1 \
--lr-warmup= \
--warmup-epoch=5 \
-j 8 \
-p 40 \
--end-epoch-validation False \
--multiplier=16 \
-b 256


# for measuring the time costs on distributed mode only
# python imagenet_training.py -a hybrid_resnet50 \
# --vanilla-arch resnet50 \
# --dist-url 'tcp://127.0.0.1:1234' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed \
# --world-size 2 \
# --rank 0 \
# ~/data/ \
# --lr 0.1 \
# --lr-decay-period 30 60 80 \
# --lr-decay-factor 0.1 \
# --mode=lowrank \
# --full-rank-warmup=False \
# --re-warmup=True \
# --fr-warmup-epoch=1 \
# --lr-warmup= \
# --warmup-epoch=5 \
# -j 8 \
# -p 40 \
# --multiplier=16 \
# -b 256


python imagenet_training.py -a hybrid_resnet50 \
--vanilla-arch resnet50 \
~/data/ \
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

# python imagenet_training.py -a lowrank_wide_resnet50_2 \
# --dist-url 'tcp://127.0.0.1:1234' \
# --dist-backend 'nccl' \
# --multiprocessing-distributed \
# --world-size 1 \
# --rank 0 \
# ~/data/ \
# --lr 0.05 \
# --lr-decay-period 30 60 80 \
# --lr-decay-factor 0.1 \
# --mode=lowrank \
# --lr-warmup= \
# --warmup-epoch=5 \
# -j 8 \
# --multiplier=16 \
# -b 256


# python imagenet_training.py -a hybrid_resnet50 \
# ~/data/ \
# --lr 0.05 \
# --lr-decay-period 30 60 80 \
# --lr-decay-factor 0.1 \
# --mode=lowarnk \
# --lr-warmup= \
# --warmup-epoch=5 \
# --multiplier=16 \
# -b 128


# python imagenet_training.py -a lowrank_resresnet50 \
# ~/data/ \
# --lr 0.1 \
# --mode=lowrank \
# --resume=checkpoint.pth.tar \
# --start-epoch=60 \
# --est-rank= \
# -b 128 > low_rank_imagenet_resnet50_b128 2>&1