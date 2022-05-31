# NeRF
This repository is code factory that I reconstructed to my research and customizing convenience.  
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)  
[Reference Code](https://github.com/yenchenlin/nerf-pytorch)  
# Demo  
![trex](https://github.com/Doyosae/NeRF/blob/main/save/trex.gif)
![horns](https://github.com/Doyosae/NeRF/blob/main/save/horns.gif)
# Start
## Dataset  
[Download Link](https://drive.google.com/file/d/1nSROtmcLvbx7xTC9lfumYhpB25zYqhuC/view?usp=sharing)  
```
Folder hierarchy
NeRF/
  /config
  /dataset
  /model_loader
  /models
  /save
  /tools
  model_train.py
  parser.py
```
## Training (only for LLFF, Blender)  
### T-rex example
```
python model_train.py --config configs/llff/trex.txt
```
### Horns example
```
python model_train.py --config configs/LLFF/horns.txt
```
## Evaluation
```
Not implemented
```
