#ESPCN
python train.py --train-file ./data/train_x2.h5 \
    --eval-file ./data/eval_x2.h5 \
    --outputs-dir ./outputs/new \
    --scale 2 \
    --lr 1e-3 \
    --batch-size 4 \
    --num-epochs 200 