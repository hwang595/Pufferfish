python train.py -data_pkl m30k_deen_shr.pkl \
-log m30k_deen_shr \
-embs_share_weight \
-proj_share_weight \
-label_smoothing \
-save_model best \
-b 256 \
-warmup 128000 \
-epoch 400 \
-seed 0 \
-fr_warmup_epoch 50 




for SEED in 1 2 3
do
  python train.py -data_pkl m30k_deen_shr.pkl \
  -log m30k_deen_shr \
  -embs_share_weight \
  -proj_share_weight \
  -label_smoothing \
  -save_model best \
  -b 256 \
  -warmup 128000 \
  -epoch 400 \
  -seed ${SEED} \
  -fr_warmup_epoch 401 > vanilla_transformer_seed${SEED}_log 2>&1
done 



# for SEED in 1 2 3
# do
#   python train.py -data_pkl m30k_deen_shr.pkl \
#   -log m30k_deen_shr \
#   -embs_share_weight \
#   -proj_share_weight \
#   -label_smoothing \
#   -save_model best \
#   -b 256 \
#   -warmup 4000 \
#   -epoch 200 \
#   -seed ${SEED} \
#   -fr_warmup_epoch 201 > vanilla_transformer_seed${SEED}_log 2>&1
# done