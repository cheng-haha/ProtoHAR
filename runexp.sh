# Local
python main.py --dataset uschad --algorithm Local --batch_size 32 --num_global_iters 300 --local_epochs 5 --sampleratio 1.0 --lamda 0.1 --learning_rate 0.001 --model cnn --personal_learning_rate 0.01 --times 1 --device 2 --niid 20k2p --total_users 14

#  FedAvg
python main.py --dataset uschad --algorithm FedAvg --batch_size 32 --num_global_iters 300 --local_epochs  5 --sampleratio 0.15 --lamda 0.1 --learning_rate 0.001 --model cnn --personal_learning_rate 0.01 --times 1 --device 2  --niid 20k2p  --slog --total_users 14 

# FedRep
python main.py --dataset uschad --algorithm FedRep --batch_size 32 --num_global_iters 300 --local_epochs  5 --sampleratio 0.15 --lamda 0.1 --learning_rate 0.001 --model cnn --personal_learning_rate 0.01 --times 1 --device 2  --niid 20k2p  --slog --total_users 14 

#  FedHome
python main.py --dataset uschad --algorithm FedHome --batch_size 32 --num_global_iters 300 --local_epochs 5 --sampleratio 0.15 --learning_rate 0.001 --model cnn --personal_learning_rate 0.01 --times 1 --device 2 --niid 20k2p --total_users 14  --fine_epochs 50

#  ProtoHAR
python main.py --dataset uschad --algorithm ProtoHAR --batch_size 32 --num_global_iters 300 --local_epochs 5 --sampleratio 0.15 --lamda 1.0 --learning_rate 0.001 --model cnn --personal_learning_rate 0.01 --times 1 --device 2 --niid 20k2p --slog --total_users 14
