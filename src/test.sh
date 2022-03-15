CUDA_VISIBLE_DEVICES=0  python -u main.py --log loguvg_2048.txt --testuvg --pretrain snapshot/2048.model --config config.json --lbda 2048
CUDA_VISIBLE_DEVICES=0  python -u main.py --log loguvg_1024.txt --testuvg --pretrain snapshot/1024.model --config config.json --lbda 1024
CUDA_VISIBLE_DEVICES=0  python -u main.py --log loguvg_512.txt --testuvg --pretrain snapshot/512.model --config config.json --lbda 512
CUDA_VISIBLE_DEVICES=0  python -u main.py --log loguvg_256.txt --testuvg --pretrain snapshot/256.model --config config.json --lbda 256
