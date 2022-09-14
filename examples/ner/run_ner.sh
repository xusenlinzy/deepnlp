# ################################## crf #############################################
# python task_sequence_labeling_cner.py \
#     --model_type=bert \
#     --model_name=crf \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --task_name=cmeee-crf \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --device_id='0' \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/crf/ \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=32 \
#     --base_model_name=bert \
#     --learning_rate=2e-5 \
#     --other_learning_rate=2e-3 \
#     --num_train_epochs=10 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --logging_steps=100 \
#     --save_steps=100 \
#     --seed=2022
  
  
# python task_sequence_labeling_cner.py \
#     --model_type=bert \
#     --model_name=cascade-crf \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --task_name=cmeee-cascade-crf \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --device_id='0' \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/cascade-crf/ \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=32 \
#     --base_model_name=bert \
#     --learning_rate=2e-5 \
#     --other_learning_rate=2e-3 \
#     --num_train_epochs=10 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --logging_steps=100 \
#     --save_steps=100 \
#     --seed=2022
  
  
# ################################## span #############################################
# python task_sequence_labeling_cner.py \
#     --task_name=cmeee-span \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/span/  \
#     --model_type=bert \
#     --model_name=span \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=32 \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --learning_rate=2e-5 \
#     --num_train_epochs=10 \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --logging_steps=100 \
#     --save_steps=100 \
#     --seed=2022
    

# ################################## gp #############################################
# python task_sequence_labeling_cner.py \
#     --model_type=bert \
#     --model_name=global-pointer \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --task_name=cmeee-global-pointer \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --device_id='0' \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/global-pointer/ \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=32 \
#     --learning_rate=2e-5 \
#     --use_rope=true \
#     --head_size=64 \
#     --head_type=global_pointer \
#     --num_train_epochs=10 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --max_grad_norm=1.0 \
#     --logging_steps=100 \
#     --save_steps=100 \
#     --seed=2022
    

# python task_sequence_labeling_cner.py \
#     --model_type=bert \
#     --model_name=global-pointer \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --task_name=cmeee-efficient-global-pointer \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --device_id='0' \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/efficient-global-pointer/ \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=32 \
#     --per_gpu_eval_batch_size=32 \
#     --learning_rate=2e-5 \
#     --use_rope=true \
#     --head_size=64 \
#     --head_type=efficient_global_pointer \
#     --num_train_epochs=10 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --max_grad_norm=1.0 \
#     --logging_steps=100 \
#     --save_steps=100 \
#     --seed=2022
    

# ################################# tplinker #############################################
# python task_sequence_labeling_cner.py \
#     --model_type=bert \
#     --model_name=tplinker \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --task_name=cmeee-tplinkerplus \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --device_id='0' \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/tplinkerplus/ \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=8 \
#     --per_gpu_eval_batch_size=8 \
#     --learning_rate=2e-5 \
#     --num_train_epochs=10 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --max_grad_norm=1.0 \
#     --logging_steps=500 \
#     --save_steps=500 \
#     --seed=2022



# #################################  mrc #############################################
# python task_sequence_labeling_cner.py \
#     --task_name=cmeee-mrc \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/mrc-cln/ \
#     --model_type=bert \
#     --model_name=mrc \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=8 \
#     --per_gpu_eval_batch_size=8 \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --use_label_embed=true \
#     --learning_rate=2e-5 \
#     --num_train_epochs=10 \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.0 \
#     --logging_steps=500 \
#     --save_steps=500\
#     --seed=2022
    

# python task_sequence_labeling_cner.py \
#     --task_name=cmeee-mrc \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/mrc-prompt/ \
#     --model_type=bert \
#     --model_name=mrc \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=8 \
#     --per_gpu_eval_batch_size=8 \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --use_label_embed=false \
#     --learning_rate=2e-5 \
#     --num_train_epochs=10 \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.0 \
#     --logging_steps=500 \
#     --save_steps=500\
#     --seed=2022


# #################################  lear #############################################
# python task_sequence_labeling_cner.py \
#     --task_name=cmeee-lear \
#     --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/lear/  \
#     --model_type=bert \
#     --model_name=lear \
#     --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
#     --do_train=true \
#     --evaluate_during_training=true \
#     --do_lower_case=true \
#     --train_max_seq_length=256 \
#     --eval_max_seq_length=256 \
#     --per_gpu_train_batch_size=16 \
#     --per_gpu_eval_batch_size=16 \
#     --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
#     --learning_rate=2e-5 \
#     --num_train_epochs=10 \
#     --checkpoint_mode=max \
#     --checkpoint_monitor=eval_f1_micro \
#     --checkpoint_save_best=true \
#     --start_thresh=0.5 \
#     --end_thresh=0.5 \
#     --gradient_accumulation_steps=1 \
#     --warmup_proportion=0.1 \
#     --logging_steps=250 \
#     --save_steps=250 \
#     --seed=2022

#################################  lear #############################################
python task_sequence_labeling_cner.py \
    --task_name=cmeee-w2ner \
    --output_dir=/home/xusenlin/nlp/deepnlp/examples/ner/outputs/cmeee/w2ner/  \
    --model_type=bert \
    --model_name=w2ner \
    --data_dir=/home/xusenlin/nlp/deepnlp/dataset/ner/cmeee \
    --do_train=true \
    --evaluate_during_training=true \
    --do_lower_case=true \
    --train_max_seq_length=256 \
    --eval_max_seq_length=256 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
    --base_model_name=bert \
    --learning_rate=1e-5 \
    --other_learning_rate=1e-3 \
    --num_train_epochs=10 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro \
    --checkpoint_save_best=true \
    --gradient_accumulation_steps=1 \
    --warmup_proportion=0.1 \
    --logging_steps=500 \
    --save_steps=500 \
    --seed=2022