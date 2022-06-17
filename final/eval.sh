python3 inference.py --gpu 0 \
--cfg rgb_only_FT \
--weight_path ./checkpoint/best.tar \
--dataset_path $1 \
--save_path $2 
wait
python3 conf.py --save_path $2