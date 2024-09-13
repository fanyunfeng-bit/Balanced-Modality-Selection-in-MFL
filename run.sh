#
python main.py --fl_method FedMDrop --global_epochs 180 --local_epochs 10 --local_lr 0.001 --lr_min 0.0001 --dataset AVE --batch_size 64 --seed 0 --client_num_per_round 5 --client_num 20 --fusion_method concat --num_frame 4 --alpha 100000000 --multi_ratio 1.0 --clientsel_algo PMR_submodular  --balansubmod_thresh 2.0 --MI_alpha 2.0
python main.py --fl_method FedMDrop --global_epochs 180 --local_epochs 10 --local_lr 0.001 --lr_min 0.0001 --dataset AVE --batch_size 64 --seed 0 --client_num_per_round 5 --client_num 20 --fusion_method concat --num_frame 4 --alpha 100000000 --multi_ratio 1.0 --clientsel_algo PMR_submodular  --balansubmod_thresh 1.5 --MI_alpha 2.0
python main.py --fl_method FedMDrop --global_epochs 180 --local_epochs 10 --local_lr 0.001 --lr_min 0.0001 --dataset AVE --batch_size 64 --seed 0 --client_num_per_round 5 --client_num 20 --fusion_method concat --num_frame 4 --alpha 3 --multi_ratio 1.0 --clientsel_algo PMR_submodular  --balansubmod_thresh 1.5 --MI_alpha 2.0
python main.py --fl_method FedMDrop --global_epochs 180 --local_epochs 10 --local_lr 0.001 --lr_min 0.0001 --dataset AVE --batch_size 64 --seed 0 --client_num_per_round 5 --client_num 20 --fusion_method concat --num_frame 4 --alpha 3 --multi_ratio 1.0 --clientsel_algo balance_submodular  --balansubmod_thresh 1.5





