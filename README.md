# VoxCNN for ADNI 3D Brain MRI classification (ISPR final project in UNIPI)

## How to run the code ?
1. Prerequisites :
```
  nvidia-docker run -v "Your local data set path":/root -it gcr.io/tensorflow/tensorflow:latest-gpu bash
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
### Result 1:
1. ROC AUC plot for AD vs Normal classification of VoxCNN

![alt text](https://github.com/lilanpei/ADNI/blob/master/Result_1_ROC%20AUC%20plot%20for%20AD%20vs%20Normal%20classification%20of%20VoxCNN.png)

2. ROC_AUC : [mean] ± [std]
```
AD vs NC : [0.87538462] ± [0.06169398]
AD vs EMCI : [0.65433333] ± [0.05580634]
AD vs LMCI : [0.76464286] ± [0.06621509]
LMCI vs NC : [0.66934524] ± [0.05654292]
LMCI vs EMCI : [0.51897321] ± [0.15433192]
EMCI vs NC : [0.49821581] ± [0.0859987]
```

3. evaluation acc : [mean] ± [std]
```
AD vs NC : [0.74743083] ± [0.1340696]
AD vs EMCI : [0.62246155] ± [0.11438376]
AD vs LMCI : [0.5398693] ± [0.08587749]
LMCI vs NC : [0.53092732] ± [0.10495938]
LMCI vs EMCI : [0.66389989] ± [0.01105777]
EMCI vs NC : [0.5365353] ± [0.04308016]
```
### Result 2:
1. ROC AUC plot for AD vs Normal classification of VoxCNN

![alt text](https://github.com/lilanpei/ADNI/blob/master/Result_2_ROC%20AUC%20plot%20for%20AD%20vs%20Normal%20classification%20of%20VoxCNN.png)

2. ROC_AUC : [mean] ± [std]
```
AD vs NC : [0.91935897] ± [0.0777566]
AD vs EMCI : [0.633] ± [0.03178312]
AD vs LMCI : [0.71464286] ± [0.07319599]
LMCI vs NC : [0.71561355] ± [0.07265784]
LMCI vs EMCI : [0.57989583] ± [0.10804529]
EMCI vs NC : [0.51877137] ± [0.09879062]
```
3. evaluation acc : [mean] ± [std]
```
AD vs NC : [0.71185771] ± [0.07849011]
AD vs EMCI : [0.66184616] ± [0.05658861]
AD vs LMCI : [0.61895427] ± [0.05988868]
LMCI vs NC : [0.60912281] ± [0.04352362]
LMCI vs EMCI : [0.60981556] ± [0.10904755]
EMCI vs NC : [0.53706441] ± [0.07779821]
```
