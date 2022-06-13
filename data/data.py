"""
    File to load dataset based on user control from main file
"""

from data.HumanPose import HumanPoseDataset

def LoadData(DATASET_NAME, DATASET_DIR, configs):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    dataset = dict()
    # handling for Human3.6M dataset
    if DATASET_NAME == 'Human36M':
        dataset['train'] = HumanPoseDataset(DATASET_NAME, DATASET_DIR, 'train', configs)
        dataset['val'] = HumanPoseDataset(DATASET_NAME, DATASET_DIR, 'val', configs)
        return dataset

    
    