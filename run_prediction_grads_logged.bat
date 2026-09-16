@echo off
cd /d "%~dp0"
if not exist prediction_gradient_outputs mkdir prediction_gradient_outputs
set LOG=prediction_gradient_outputs\run_log.txt
set PYTHONIOENCODING=utf-8
echo START %date% %time% > %LOG%
echo ===== python visualize_prediction_gradients.py --dataset mnist --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset mnist --num_images 10 >> %LOG% 2>&1
echo ===== python visualize_prediction_gradients.py --dataset fashion_mnist --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset fashion_mnist --num_images 10 >> %LOG% 2>&1
echo ===== python visualize_prediction_gradients.py --dataset cifar10 --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset cifar10 --num_images 10 >> %LOG% 2>&1
echo ===== python visualize_prediction_gradients.py --dataset cifar10 --model robustbench:cifar10:Carmon2019Unlabeled --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset cifar10 --model robustbench:cifar10:Carmon2019Unlabeled --num_images 10 >> %LOG% 2>&1
echo ===== python visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model resnet50 --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model resnet50 --num_images 10 >> %LOG% 2>&1
echo ===== python visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model squeezenet1_1 --num_images 10 >> %LOG%
python -u visualize_prediction_gradients.py --dataset imagenet --data_root .\imagenet_samples --model squeezenet1_1 --num_images 10 >> %LOG% 2>&1
echo ALLDONE %date% %time% >> %LOG%
