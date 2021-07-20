python /home/ubuntu/low-rank-ml/early_bird_ticket/resprune_50.py \
--dataset imagenet \
--data /home/ubuntu/data \
--arch resnet50_official \
--test-batch-size 128 \
--depth 50 \
--percent 0.7 \
--model /home/ubuntu/low-rank-ml/early_bird_ticket/scripts/resnet50-imagenet/EBTrain-ImageNet/ResNet50/EB-70-9.pth.tar \
--save /home/ubuntu/low-rank-ml/early_bird_ticket/scripts/resnet50-imagenet/EBTrain-ImageNet/ResNet50/pruned_7008_0.7