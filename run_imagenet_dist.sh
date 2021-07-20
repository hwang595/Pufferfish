python imagenet_training.py -a hybrid_resnet50 \
/home/ubuntu/data/ \
--lr 0.1 \
-b 128 \
--multiprocessing-distributed \
--dist-backend 'nccl' \
--dist-url 'tcp://172.31.10.54:1234' \
--mode=lowrank \
--world-size 2 \
--lr-decay-period 30 60 80 \
--full-rank-warmup=True \
--fr-warmup-epoch=2 \
--lr-warmup= \
--re-warmup=True \
--warmup-epoch=5 \
-j 8 \
-p 80 \
--model-save-dir /home/ubuntu/low-rank-ml \
--rank 0