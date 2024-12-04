export XLA_FLAGS='--xla_gpu_deterministic_ops=true' 
export TF_DETERMINISTIC_OPS=1
python experiments/$1/main.py --settings="experiments/$1/settings.json" | tee experiments/$1/output.txt
