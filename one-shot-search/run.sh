#python train.py --task_dir=../KG_Data/FB15K237 --model=simple --n_epoch=200 --epoch_per_test=40 --test_batch_size=50 --out_file=_tune;
#python large.py --task_dir=../KG_Data/YAGO --model=simple --n_epoch=400 --epoch_per_test=40 --test_batch_size=80 --out_file_info=_large --gpu=1;

python evaluate.py --task_dir=../KG_Data/WN18 --gpu=6;
python evaluate.py --task_dir=../KG_Data/FB15K --gpu=6;
python evaluate.py --task_dir=../KG_Data/WN18RR --gpu=6;
python evaluate.py --task_dir=../KG_Data/FB15K237 --gpu=6;
