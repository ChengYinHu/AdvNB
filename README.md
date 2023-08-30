# AdvNB
Code and model for "Adversarial Neon Beam: A Light-based Physical Attack to DNNs" (ACM MM 2023)
<p align='center'>
  <img src='dataset/p1.jpg' width='700'/>
</p>

## Introduction
In the physical world, especially in busy cities, there are so many neon beams that they tend to scatter on traffic signs, causing humans to instinctively ignore them. If an attacker deliberately creates an adversarial neon beam that can attack self-driving car systems while lowering human vigilance, it could disrupt traffic.
In this work, we propose a light-based physical attack called adversarial neon beam(AdvNB).which enables the manipulations of neon beam's physical parameters to perform effective physical attacks to advanced DNNs.

## Requirements
* python == 3.8
* torch == 1.8.0

## Basic Usage
```sh
python digital_test.py --model resnet50 --dataset your_dataset
```

