# imagenet signsum
NUM_NODES=16
RANK=0

python main_clean_up.py \
--model-type imagenet \
--norm-file "log_file.log" \
--compressor signum \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.26.2:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK} > log_rank_${RANK} 2>&1


# cifar10 signum
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
--rank ${RANK} > log_rank_${RANK} 2>&1




NUM_NODES=16
RANK=0

python main_clean_up.py \
--model-type imagenet \
--norm-file "log_file.log" \
--compressor vanilla \
--config-mode pufferfish \
--distributed \
--master-ip 'tcp://172.31.4.231:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK} > log_rank_${RANK} 2>&1