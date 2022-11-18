import numpy as np

def pck(annotations, predictions, torso_indices, t=0.1):
    torso_keypoints = annotations[:, torso_indices]
    torso_distances = np.linalg.norm(torso_keypoints[:, 0] - torso_keypoints[:, 1], axis=1)
    kp_differences = annotations - predictions
    kp_distances = np.linalg.norm(kp_differences, axis=2)
    kp_distances = (kp_distances.transpose() - torso_distances * t).transpose()  # broadcasting is only possible in the last dimension
    correct_keypoints = kp_distances <= 0
    pck_joint = np.sum(correct_keypoints, axis=0) / correct_keypoints.shape[0]
    pck_all = np.sum(correct_keypoints) / correct_keypoints.shape[0] / correct_keypoints.shape[1]
    return pck_all, pck_joint

