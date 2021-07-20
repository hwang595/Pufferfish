python main.py \
--cuda \
--lr 20 \
--emsize 1500 \
--nhid 1500 \
--dropout 0.65 \
--epochs 40 \
--warmup True \
--warmup-epoch 5 \
--tied > low_rank_lstm_dim1500_log3 2>&1


for SEED in 1 2 3
do
  python main.py \
  --cuda \
  --lr 20 \
  --emsize 1500 \
  --nhid 1500 \
  --dropout 0.65 \
  --epochs 40 \
  --warmup True \
  --seed ${SEED} \
  --warmup-epoch 10 \
  --tied > pufferfish_lstm_dim1500_seed${SEED}_log 2>&1
done

# for evaluation
python main.py \
--cuda \
--lr 20 \
--emsize 1500 \
--nhid 1500 \
--dropout 0.65 \
--epochs 40 \
--warmup True \
--seed 1 \
--evaluate True \
--ckpt-path /home/ubuntu/low-rank-ml/word_language_model/pufferfish_model_lowrank_seed1_best.pt \
--warmup-epoch 5 \
--tied 

# for training
for SEED in 1 2 3
do
  python main.py \
  --cuda \
  --lr 20 \
  --emsize 1500 \
  --nhid 1500 \
  --dropout 0.65 \
  --epochs 40 \
  --warmup True \
  --seed ${SEED} \
  --warmup-epoch 41 \
  --tied > vanilla_lstm_dim1500_seed${SEED}_log 2>&1
done