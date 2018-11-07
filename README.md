# VoxCNN for ADNI 3D Brain MRI classification

## How to run the code ?
1. Prerequisites :
```
  pip install --upgrade nibabel
  pip install --upgrade keras==2.1
  pip install --upgrade setGPU
```
2. Set data set and log path :
```
  ds_path = Your data set path
  plot_history_path = Your log path
```
3. Start training :
```
  python ADNI.py
```
4. Calculate results and plotting :
```
  python Plot.py
```
