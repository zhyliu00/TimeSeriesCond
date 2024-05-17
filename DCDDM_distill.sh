init='real'
gpu=0
threads=32
dataset=har
ipc=5
Iteration=2000
eval_it=200
epoch_eval_train=300
num_eval=5
framework='DCDDM'
lr_feat=1
lr_teacher=1e-3
lr_lr=1e-6
# lr_lr=0
lambda_DM=1
model='CNNBN'
max_start_epoch=60
expert_epochs=40
syn_steps=10
inputaug='raw_LPF_FTPP_FTMP'

OMP_NUM_THREADS=${threads} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u DCDDM_distill.py --inputaug $inputaug --lambda_DM $lambda_DM --framework $framework  --model $model --eval_it $eval_it --dataset $dataset --num_eval $num_eval --epoch_eval_train $epoch_eval_train --pix_init $init --Iteration $Iteration --syn_steps $syn_steps --ipc $ipc --max_start_epoch $max_start_epoch --expert_epochs $expert_epochs  --lr_lr $lr_lr --lr_teacher $lr_teacher --lr_feat ${lr_feat} > ./out/${dataset}/DCDDM_distill_verbose_${dataset}_ipc${ipc}_init${init}_lr${lr_feat}_gpu${gpu}.out 2>&1 &
