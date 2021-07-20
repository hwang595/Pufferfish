NUM_NODES=2
RANK=0

python main_simulation.py \
--model-type imagenet \
--norm-file "log_file.log" \
--compressor signum \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.26.2:1234' \
--num-simulated-nodes 4 \
--num-nodes ${NUM_NODES} \
--rank ${RANK} > log_rank_${RANK} 2>&1