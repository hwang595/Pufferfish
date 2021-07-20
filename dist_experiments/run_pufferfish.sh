# imagenet pufferfish
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type imagenet \
--norm-file "log_file.log" \
--compressor vanilla \
--config-mode pufferfish \
--distributed \
--master-ip 'tcp://172.31.13.181:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}


# cifar10 pufferfish
NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor vanilla \
--config-mode pufferfish \
--distributed \
--master-ip 'tcp://172.31.13.181:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}