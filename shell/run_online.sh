# CIFAR10 VGG16
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
python refer_model_online.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 256 --state victim
python refer_model_online.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 256 --state shadow
# python refer_model_online.py config/cifar10/cifar10_vgg16.json --distributed True --world_size 4 --model_num 256 --state victim
# python refer_model_online.py config/cifar10/cifar10_vgg16.json --distributed True --world_size 4 --model_num 256 --state shadow
python mia_attack_online.py 0 config/cifar10/cifar10_vgg16.json --model_num 256 --attack_rapid
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack