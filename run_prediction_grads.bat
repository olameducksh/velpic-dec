@echo on
cd /d "%~dp0"
python visualize_prediction_gradients.py --dataset mnist --num_images 10
python visualize_prediction_gradients.py --dataset fashion_mnist --num_images 10
python visualize_prediction_gradients.py --dataset cifar10 --num_images 10
python visualize_prediction_gradients.py --dataset cifar10 --model robustbench:cifar10:Carmon2019Unlabeled --num_images 10
python visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model resnet50 --num_images 10
python visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model squeezenet1_1 --num_images 10
pause
