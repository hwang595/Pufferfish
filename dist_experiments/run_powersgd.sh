node_rank=0

python main.py \
--model-type imagenet \
--fixed-sched \
--norm-file "wideresnet_log_file_powersgd_k4.log" \
--start-k \
--k-start 4 \
--distributed \
--master-ip "tcp://172.31.3.12:1234" \
--num-nodes 4 \
--rank ${node_rank} > log_file_node${node_rank} 2>&1