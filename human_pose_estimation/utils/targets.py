import numpy as np


def create_heatmaps(joints, output_size, visibility,  sigma=2):
    size = 6 * sigma + 3
    offset = 3 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    joints = cut_coords(joints, output_size, 0)

    # create heatmap in a way that the gaussian blobs fit into it even if the keypoint is at the border: make it bigger
    heatmaps = np.zeros((joints.shape[0], output_size + size, output_size + size))

    # [0, ..., 0, ..., (num_joints-1), ..., (num_joints-1)] (size * size times)
    indices_joints = np.transpose(np.tile(np.arange(joints.shape[0]), (size ** 2, 1)), (1, 0)).flatten()
    # [[0, 1, ..., (size-1), 0, ..., (size-1)], ..., [...]] (num_joints times)
    cnt_indices = np.tile(np.arange(size), (joints.shape[0] * size, 1)).reshape((joints.shape[0], size ** 2))
    # size times x_(joint 0) + [0, 1, ..., (size-1)] then size times x_(joint 1) + [0, 1, ..., (size-1)] etc.
    indices_x = np.asarray((np.tile(joints[:, 0].flatten(), (size * size, 1)).transpose((1, 0)) + cnt_indices).flatten(), dtype=np.int32)
    # [[0, 1, ..., (size-1)], ..., [0, 1, ..., (size-1)]] num_joints times
    cnt_indices = np.tile(np.arange(size), (joints.shape[0], 1))
    # [size times y_(joint 0) + 0, size times y_(joint 0) + 1, ..., size times y_(joint 0) + (size-1), ..]
    indices_y = np.asarray(np.transpose(np.tile(np.transpose(np.tile(joints[:, 1].flatten(), (size, 1)), (1, 0)) + cnt_indices, (size, 1, 1)), (1, 2, 0)).flatten(), dtype=np.int32)

    # flatten g and repeat num_joints times
    g = np.tile(g.flatten(), joints.shape[0])
    # put g at correct positions
    heatmaps[indices_joints, indices_y, indices_x] = g
    # reduce size of heatmaps, we added a padding so that gaussian blobs fit also if keypoints are at the border, we remove the padding now
    heatmaps = heatmaps[:, offset:-(size - offset), offset:-(size - offset)]
    heatmaps = np.asarray(heatmaps, dtype=np.float32)

    # remove non visible joints
    heatmaps = np.where(np.expand_dims(np.expand_dims(visibility, axis=-1), axis=-1) == 1, heatmaps, np.zeros_like(heatmaps))

    return heatmaps


def cut_coords(coords, size, blob_width):
    coords[:, :2] = np.asarray(coords[:, :2], dtype=np.int32)  # rounding might be better, but let's stay close to the original code
    coords[coords[:, 0] >= size + blob_width] = 0
    coords[coords[:, 0] < - blob_width] = 0
    coords[coords[:, 1] >= size + blob_width] = 0
    coords[coords[:, 1] < - blob_width] = 0
    return coords
