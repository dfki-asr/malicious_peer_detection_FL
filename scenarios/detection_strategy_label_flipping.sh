strategy='detection_strategy'
attack='label_flipping'

# download dataset
python3 utils/dl_dataset.py --dataset mnist

# generate partitions
python3 utils/partition_data.py --n_partitions 10 --dataset mnist

# start server and clients
python3 server.py --strategy $strategy --attack $attack &
sleep 3
python3 client.py --num 0 --attack 'none' &
python3 client.py --num 1 --attack $attack