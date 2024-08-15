# Socialized Learning: Making Each Other Better Through Multi-Agent Collaboration (SL)

The code repository for "Socialized Learning: Making Each Other Better Through Multi-Agent Collaboration" (the paper has been accepted by ICML 2024) in PyTorch.

## Prerequisites

The following packages are required to run the scripts:

Please see [requirements.txt](./requirements.txt) and [env.yaml](./env.yaml)

## Dataset
We provide the benchmark dataset, i.e., [CIFAR100](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf). 

Each dataset can be referred to the following link. If the link cannot be accessed, it can be directly obtained in the ./datasets/ provided by us.

CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html

## Testing scripts

- Test CIFAR100 (MASC)
  ```
  python test_multi_agent_cifar100_ts.py --batch_size 10000 --epochs 1 --num_works_multi_task 100 --num_works_multi_agent 100 --path ./datasets/data_cifar100_raw --model_path_s ./checkpoints/student_checkpoint/task_100 --model_path_t ./checkpoints/teacher_0_checkpoint/task_0 --gpu 0
  ```
  
  Remember to change `--path`, `--model_path_s`, and `--model_path_t` into your own root, or you will encounter errors.

- Test CIFAR100 (Teacher)
  ```
  python test_single_agent_cifar100.py --batch_size 10000 --epochs 1 --num_works_single_task 0 --num_works_single_agent 0 --path ./datasets/data_cifar100_raw --model_path ./checkpoints/before_teacher_0_checkpoint/task_0 --gpu 0
  ```
  Remember to change `--path`, and `--model_path` into your own root, or you will encounter errors.

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [Learngene: From Open-World to Your Learning Task](https://github.com/BruceQFWang/learngene)
- [Isolation and Impartial Aggregation: A Paradigm of Incremental Learning without Interference](https://github.com/iamwangyabin/ESN?utm_source=catalyzex.com)

## Contact 
If there are any questions, please feel free to contact with the author:  Xinjie Yao (yaoxinjie@tju.edu.cn). Enjoy the code.
