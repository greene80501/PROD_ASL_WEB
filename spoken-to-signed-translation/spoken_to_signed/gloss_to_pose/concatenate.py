from typing import List

import numpy as np
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, correct_wrists, pose_normalization_info, normalize_pose_size

from smoothing import smooth_concatenate_poses

class ConcatenationSettings:
    is_reduce_holistic = True


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose_normalization_info(pose.header))


def trim_pose(pose, start=True, end=True):
    if len(pose.body.data) == 0:
        raise ValueError("Cannot trim an empty pose")

    wrist_indexes = [
        pose.header._get_point_index('LEFT_HAND_LANDMARKS', 'WRIST'),
        pose.header._get_point_index('RIGHT_HAND_LANDMARKS', 'WRIST')
    ]
    either_hand = pose.body.confidence[:, 0, wrist_indexes].sum(axis=1) > 0

    first_non_zero_index = np.argmax(either_hand) if start else 0
    last_non_zero_index = (len(either_hand) - np.argmax(either_hand[::-1])) if end else len(either_hand)

    pose.body.data = pose.body.data[first_non_zero_index:last_non_zero_index]
    pose.body.confidence = pose.body.confidence[first_non_zero_index:last_non_zero_index]
    return pose


def concatenate_poses(poses: List[Pose], trim=True) -> Pose:
    if ConcatenationSettings.is_reduce_holistic:
        print('Reducing poses...')
        poses = [reduce_holistic(p) for p in poses]

    print('Normalizing poses...')
    poses = [normalize_pose(p) for p in poses]

    # Trim the poses to only include the parts where the hands are visible
    if trim:
        print('Trimming poses...')
        poses = [trim_pose(p, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

    # Concatenate all poses
    print('Smooth concatenating poses...')
    pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print('Correcting wrists...')
    pose = correct_wrists(pose)

    # Scale the newly created pose
    print('Scaling pose...')
    normalize_pose_size(pose)

    return pose
