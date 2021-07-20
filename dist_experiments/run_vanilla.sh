# imagenet vanilla
NUM_NODES=2
RANK=0

python main_clean_up.py \
--model-type imagenet \
--norm-file "log_file.log" \
--compressor vanilla \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.26.2:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}



# cifar10 vanilla
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
--rank ${RANK} > log_rank_${RANK} 2>&1