gpu=0
threads=32
dataset='har'
# model='LSTM'
# model='GRU'
# model='CNNIN'
# model='ResNet18BN'
# model='Transformer'
model='CNNBN'

lr=0.0001
train_epochs=100
num_experts=50
batch_train=512
inputaug='raw_LPF_FTPP_FTMP'

OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u DCDDM_buffer.py --inputaug $inputaug --lr_teacher $lr --model $model --batch_train $batch_train --dataset ${dataset} --train_epochs $train_epochs --num_experts $num_experts > ./out/${dataset}/train_verbose_${dataset}_${model}_aug${aug}_gpu${gpu}.out 2>&1 &
