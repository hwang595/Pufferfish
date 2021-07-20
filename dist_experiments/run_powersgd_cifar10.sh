NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor powersgd \
--config-mode vanilla \
--distributed \
--master-ip 'tcp://172.31.6.30:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}