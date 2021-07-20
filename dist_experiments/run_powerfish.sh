NUM_NODES=8
RANK=0

python main_clean_up.py \
--model-type CNN \
--norm-file "log_file.log" \
--compressor powersgd \
--config-mode powerfish \
--distributed \
--master-ip 'tcp://172.31.0.8:1234' \
--num-nodes ${NUM_NODES} \
--rank ${RANK}