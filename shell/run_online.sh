# CIFAR10 VGG16
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
python refer_model_online.py 0 config/cifar10/cifar10_vgg16.json --model_num 64 --state victim
python refer_model_online.py 0 config/cifar10/cifar10_vgg16.json --model_num 64 --state shadow
python mia_attack_online.py 0 config/cifar10/cifar10_vgg16.json --model_num 64 --query_num 8
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack