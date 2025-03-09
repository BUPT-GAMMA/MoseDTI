# MoseDTI
The official implementation of AAAI25 paper "Blend the Separated: Mixture of Synergistic Experts for Data-Scarcity Drug-Target Interaction Prediction".

[Read the paper](http://www.shichuan.org/doc/187.pdf)

![image](https://github.com/user-attachments/assets/5400104b-7b78-4f35-8852-97a8971a86bb)

## Usage

1. Unzip the esm.tar.gz to your anaconda envs directory and activate the esm environment.
2. Execute the split_data.py to split the data as cross-validation splits.
3. Train model from a pretrained KGE model: 
   ```
   python kge/std_main.py --dataset ago_10shots_0 --device 5 --seed 1 --load_kge_model 2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth
   ```
   You can also train the KGE model yourself without the --load_kge_model argument. You can also save the intermediate models with the --save and load them with the --load* arguments.

If there are any issues or cooperation intentions, please contact zhaijojo@bupt.edu.cn.
