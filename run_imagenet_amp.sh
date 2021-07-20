python amp_imagenet_training.py -a amp_hybrid_resnet50 \
--vanilla-arch amp_resnet50 \
~/data/ \
--lr 0.1 \
--lr-decay-period 30 60 80 \
--lr-decay-factor 0.1 \
--mode=lowrank \
--full-rank-warmup=True \
--re-warmup=True \
--fr-warmup-epoch=10 \
--lr-warmup= \
--warmup-epoch=5 \
-j 16 \
-p 40 \
--multiplier=16 \
-b 256