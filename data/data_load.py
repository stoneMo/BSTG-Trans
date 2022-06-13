import numpy as np


def load_data(output_filename_3d, output_filename_2d, mode, input_length=5, label_length=5):

    pose_array_3d = np.load(output_filename_3d, allow_pickle=True)
    pose_array_2d = np.load(output_filename_2d, allow_pickle=True)

    subjects = list(pose_array_3d['positions_3d'].item().keys())

    data_input_list = []
    data_label_list = []

    if mode == 'train':
        subjects.remove('S5')
        # print("subjects:", subjects)
    elif mode == 'val':
        subjects = ['S5']

    print("subjects:", subjects)
    
    for subj in subjects:

        pose_3d_subj = pose_array_3d['positions_3d'].item()[subj]
        pose_2d_subj = pose_array_2d['positions_2d'].item()[subj]

        action_list = pose_3d_subj.keys()

        for action in action_list:

            if action in pose_2d_subj:

                pose_3d = pose_3d_subj[action]               # (1383, 32, 3)
                # pose_3d = pose_3d_subj[action]                # (1383, 17, 2)
                pose_2d = pose_2d_subj[action][0]            # (1383, 17, 2)

                num_pose = min(len(pose_3d), len(pose_2d))
                
                for i in range(num_pose):

                    if i + input_length < num_pose and i + input_length + label_length < num_pose:

                        input = pose_2d[i:i+input_length]
                        label = pose_3d[i+input_length:i+input_length+label_length]
                    
                        data_input_list.append(input)
                        data_label_list.append(label)
    
    return data_input_list, data_label_list

if __name__ == "__main__":

    output_filename_3d = './human36/data_3d_h36m.npz'
    output_filename_2d = './human36/data_2d_h36m_gt.npz'

    mode = 'train'

    data_input_list, data_label_list = load_data(output_filename_3d, output_filename_2d, mode)

    print("data_input_list:", data_input_list[0].shape)            # train: 424905, val: 98779
    print("data_label_list:", data_label_list[0].shape)