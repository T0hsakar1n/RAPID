# CIFAR10 VGG16
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
python refer_model.py config/cifar10/cifar10_vgg16.json --device 0 --model_num 8
# python refer_model.py config/cifar10/cifar10_vgg16.json --distributed True --world_size 4 --model_num 8
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json  --attack_rapid
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack

