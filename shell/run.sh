# CIFAR10 VGG16
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
python refer_model.py 0 config/cifar10/cifar10_vgg16.json --model_num 4
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack

