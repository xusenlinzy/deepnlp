################################## gplinker #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=gplinker \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-gplinker \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/gplinker/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --use_pfn=true \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42
  
  
################################## casrel #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=casrel \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-casrel \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/casrel/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42
    

################################## grte #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=grte \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-grte \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/grte/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=4000 \
  --save_steps=4000 \
  --seed=42
    

################################# tplinker #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=tplinker \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-tplinkerplus \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/tplinkerplus/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42



#################################  spn #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=spn \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-spn \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/spn/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --na_rel_coef=0.25 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42
  

#################################  prgc #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=prgc \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-prgc \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/prgc/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42


################################## pfn #############################################
python task_relation_extraction.py \
  --model_type=bert \
  --model_name=pfn \
  --pretrained_model_path='hfl/chinese-roberta-wwm-ext' \
  --task_name=duie-pfn \
  --do_train=true \
  --evaluate_during_training=true \
  --do_lower_case=true \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --checkpoint_save_best=true \
  --data_dir=/home/xusenlin/nlp/deepnlp/dataset/re/duie \
  --output_dir=/home/xusenlin/nlp/deepnlp/examples/re/outputs/duie/pfn/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --decode_thresh=0.5 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.0 \
  --logging_steps=2000 \
  --save_steps=2000 \
  --seed=42