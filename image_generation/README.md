# f-EBM

Code for [Training Deep Energy-Based Models with f-Divergence Minimization](https://arxiv.org/pdf/2003.03463.pdf).

## Setup
Install the prerequisites:
```
pip install -r requirements.txt
mkdir cachedir
```

Pretrained models can be downloaded [here](https://drive.google.com/file/d/183zXzbIu_cY6r7eb_aQ3glWxHo9Tc1ML/view?usp=sharing). 
Unzip them to cachedir (e.g. `cachedir/cifar10_cond_febm_new_jensen_shannon`). Modify the dataset paths in `data.py`.

The loss function of f-EBM is implemented [here](https://github.com/ermongroup/f-EBM/blob/master/image_generation/train_febm.py#L730-L739). See Appendix E.1 in the paper for discussions on the implementations.

## Training and Testing:

To train on CIFAR-10 dataset using 4 GPUs:
```
mpiexec --oversubscribe -n 4 python train_febm.py --exp=cifar10_cond_febm_new --dataset=cifar10 --divergence=jensen_shannon --num_steps=60 --batch_size=128 --step_lr=10.0 --proj_norm=0.01 --zero_kl --replay_batch --cclass
```
To use different f-divergences, modify `--divergence=jensen_shannon/squared_hellinger/reverse_kl/kl`.

To test the models:
```
python test_inception_febm.py --logdir=cachedir --exp=cifar10_cond_febm_new_jensen_shannon --batch_size=512 --resume_iter=45900 --cclass --im_number=50000 --step_lr=10.0 --num_steps=5 --repeat_scale=180 --nomix=140 --ensemble=1
```
Output: Inception score of 8.606776382446289 with std of 0.05907025933265686 FID of score 30.863331867437637

```
python test_inception_febm.py --logdir=cachedir --exp=cifar10_cond_febm_new_squared_hellinger --batch_size=512 --resume_iter=79300 --cclass --im_number=50000 --step_lr=10.0 --num_steps=5 --repeat_scale=180 --nomix=140 --ensemble=1
```
Output: Inception score of 8.572072982788086 with std of 0.08429072052240372 FID of score 32.1934893353108

```
python test_inception_febm.py --logdir=cachedir --exp=cifar10_cond_febm_new_reverse_kl --batch_size=512 --resume_iter=67900 --cclass --im_number=50000 --step_lr=10.0 --num_steps=5 --repeat_scale=180 --nomix=140 --ensemble=1
```
Output: Inception score of 8.487509727478027 with std of 0.08897720277309418, FID of score 33.25769397031337

```
python test_inception_febm.py --logdir=cachedir --exp=cifar10_cond_febm_new_kl --batch_size=512 --resume_iter=74800 --cclass --im_number=50000 --step_lr=10.0 --num_steps=5 --repeat_scale=180 --nomix=140 --ensemble=1
```
Output: Inception score of 8.105894638061523 with std of 0.06555138528347015, FID of score 37.36842518619477

## Acknowledgement
The implementation is based on this repository: https://github.com/openai/ebm_code_release.