# BSTG-Trans: A Bayesian Spatial-Temporal Graph Transformer for Long-term Pose Forecasting

Human pose forecasting that aims to predict the body poses happening in the future is an important task in computer vision. However, long-term pose forecasting is particularly challenging because modeling long-range dependencies across the spatial-temporal level is hard for joint-based poses. Another challenge is uncertainty prediction since the future prediction is not a deterministic process. In this work, we present a novel **B**ayesian **S**patial-**T**emporal **G**raph **Trans**former (BSTG-Trans) for predicting accurate, diverse, and uncertain future poses. First, we apply a spatial-temporal graph transformer as an encoder and a temporal-spatial graph transformer as a decoder for modeling the long-range spatial-temporal dependencies across pose joints to generate the long-term future body poses. Furthermore, we propose a Bayesian sampling module for uncertainty quantization of diverse future poses. Finally, a novel uncertainty estimation metric, namely Uncertainty Absolute Error (UAE) is introduced for measuring both the accuracy and uncertainty of each predicted future pose. We achieve state-of-the-art performance against other baselines on Human3.6M and HumanEva-I in terms of accuracy, diversity, and uncertainty for long-term pose forecasting. Moreover, our comprehensive ablation studies demonstrate the effectiveness and generalization of each module proposed in our BSTG-Trans. 


![tenser](figures/figures/framework_bstg_trans.png)


## Environment Setup

Please install the requirements, run
```
pip install -r requirements.txt
```

## Data Preparation

Please follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. Place the prepocessed data ``data_3d_h36m.npz`` and ``data_2d_h36m_gt.npz`` under the ``data`` folder.


## Model 

A trained model **checkpoint.pkl** is provided in [CHECKPOINT](https://drive.google.com/file/d/1MO2guGhsczS7VQwWfRt3086sntx7qeCp/view?usp=sharing).


## Running Instructions

To perform experiments with BSTG-Trans, run: 
```
python main_body_pose.py --dataset Human36M \
    --batch_size 128 \
    --init_lr 0.001 \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --seed 2022 \
    --config 'configs/bayesian_stgt_human36m.json'
```

