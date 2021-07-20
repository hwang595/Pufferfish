node_rank=0

python main.py \
--model-type imagenet \
--fixed-sched \
--norm-file "eb_train_30p.log" \
--start-k \
--distributed \
--master-ip 'tcp://172.31.32.175:1234' \
--num-nodes 8 \
--rank ${node_rank} > log_file_node${node_rank} 2>&1