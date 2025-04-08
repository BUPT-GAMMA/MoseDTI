# MoseDTI
The official implementation of AAAI25 paper "Blend the Separated: Mixture of Synergistic Experts for Data-Scarcity Drug-Target Interaction Prediction".

2025/4 📢 News：For easier env configuration, we have separately provided the environment_conda_env_export.yml output by "conda env export" and requirements_pip_freeze.txt output by "pip freeze".
2025/3 📢 News：We have provided the whole dataset and the pretrained kge model (a part of our whole model, which can be reused for every dataset as they use the same KG).

[Read the paper](http://www.shichuan.org/doc/187.pdf)

![image](https://github.com/user-attachments/assets/5400104b-7b78-4f35-8852-97a8971a86bb)

## Usage
1. Download large files. Put [drkg.tsv](https://drive.google.com/file/d/1-pRYRtgcNFqxeL3Q9ZgxIU5rnJbfKB8M/view?usp=sharing) into var_data/ and put [kge model](https://drive.google.com/file/d/1_RCRrHJBosWycpqzxXxzrmJYCGWaJfc8/view?usp=sharing) into a new folder var_models/. Configure your conda environment according to requirements.txt output by "conda list --export", or the environment_conda_env_export.yml output by "conda env export" and requirements_pip_freeze.txt output by "pip freeze".
2. Execute the split_data.py to split the data as cross-validation splits.
3. Train and evaluate model from a pretrained KGE model: 
   ```
   python kge/std_main.py --dataset ago_10shots_0 --device 0 --load_kge_model 2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth
   ```
   You can also train the KGE model yourself without the --load_kge_model argument. You can also save the model components with the --save and load them with the --load* arguments.

If there are any issues or cooperation intentions, please contact zhaijojo@bupt.edu.cn.

## Citation
```
@inproceedings{zhai2025mosedti,
    title={Blend the Separated: Mixture of Synergistic Experts for Data-Scarcity Drug-Target Interaction Prediction},
    author={Zhai, Xinlong and Wang, Chunchen and Wang, Ruijia and Kang, Jiazheng and Li, Shujie and Chen, Boyu and Ma, Tengfei and Zhou, Zikai and Yang, Cheng and Shi, Chuan},
    booktitle={Association for The Advancement of Artificial Intelligence},
    year={2025}
}
```
