import rosbag
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import rospy
import sensor_msgs.point_cloud2 as pc2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import List


def get_lidar_to_world(cam2cam0):
    lidar2cam = np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                            [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                            [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
                            [0, 0, 0, 1]])
    cam2lidar = np.linalg.inv(lidar2cam)
    lidar2lidar0 = np.dot(np.dot(cam2lidar, cam2cam0), lidar2cam)
    return lidar2lidar0

def create_bag(kitti_path, seq=0):
    bridge = CvBridge()

    seq = str(seq).zfill(2)
    image_dir = os.path.join(kitti_path, seq, 'image_2')
    velodyne_dir = os.path.join(kitti_path, seq, 'velodyne')
    timestamps_path = os.path.join(kitti_path, seq, 'times.txt')
    pose_path = os.path.join(kitti_path, '../poses', f"{seq}.txt")
    new_pose_path = os.path.join(kitti_path, seq, 'ground_truth_traj.txt')

    with open(timestamps_path, 'r') as f:
        timestamps = f.readlines()

    with open(pose_path, 'r') as f:
        poses = f.readlines()

    pose_file = open(new_pose_path, 'w')

    bag_path = os.path.join(kitti_path, seq, 'data.bag')
    bag = rosbag.Bag(bag_path, 'w')

    # write velodyne
    for idx, timestamp in enumerate(tqdm(timestamps, desc='Writing velodyne of sequence {}'.format(seq))):
        timestamp = float(timestamp.strip())
        timestamp = rospy.Time.from_sec(timestamp)

        # pose: matrix 3x4
        pose = [float(p) for p in poses[idx].strip().split()]
        pose = np.array(pose).reshape(3, 4)
        pose_mat = np.eye(4)
        pose_mat[:3, :] = pose

        pose_mat = get_lidar_to_world(pose_mat)

        # "timestamp","x","y","z","q_x","q_y","q_z","q_w"
        new_pose = np.zeros((8,))
        new_pose[0] = timestamp.to_sec()
        new_pose[1:4] = pose_mat[:3, 3]
        q = R.from_matrix(pose_mat[:3, :3]).as_quat()
        new_pose[4:] = q
        new_pose = [str(p) for p in new_pose]
        new_pose = ' '.join(new_pose)
        pose_file.write(new_pose + '\n')


        velodyne_path = os.path.join(velodyne_dir, '{:06d}.bin'.format(idx))
        velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        if velodyne is None:
            continue

        velodyne_msg = pc2.create_cloud_xyz32(header=rospy.Header(frame_id='velodyne'), points=velodyne[:, :3])
        velodyne_msg.header.stamp = timestamp
        velodyne_msg.header.frame_id = "velodyne"

        bag.write('/velodyne_points', velodyne_msg, t=timestamp)

    # write image
    # for idx, timestamp in enumerate(timestamps):
    #     timestamp = float(timestamp.strip())
    #     image_path = os.path.join(image_dir, '{:06d}.png'.format(idx))
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if image is None:
    #         continue

    #     image_msg = bridge.cv2_to_imgmsg(image, encoding="mono8")
    #     image_msg.header.stamp = rospy.Time.from_sec(timestamp)
    #     image_msg.header.frame_id = "camera"

    #     bag.write('/camera/image_raw', image_msg, t=timestamp

    bag.close()
    pose_file.close()

def mt_wrapper(func, args_list, chunk_size=30000, parallel_mtd="none"):
    # split the args_list into chunks to avoid memory error and show progress bar
    if not isinstance(args_list[0], tuple):
        args_list = [(x,) for x in args_list]
    results = []
    # get the name of the function
    name = func.__name__ if hasattr(func, "__name__") else func.func.__name__
    # timer.tic(name)
    if parallel_mtd == "none":
        for i in tqdm(
            range(0, len(args_list)), desc="Execute {}".format(name), leave=False
        ):
            results.append(func(*args_list[i]))
    elif parallel_mtd == "thread":
        chunk_size = min(chunk_size, len(args_list))
        for i in tqdm(
            range(0, len(args_list), chunk_size),
            desc="Multi-thread {} {}".format(name, chunk_size),
            leave=False,
        ):
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(func, *args)
                    for args in args_list[i : i + chunk_size]
                ]
                results.extend([future.result() for future in futures])
    elif parallel_mtd == "process":
        chunk_size = min(chunk_size, len(args_list))
        for i in tqdm(
            range(0, len(args_list), chunk_size),
            desc="Multi-process {} {}".format(name, chunk_size),
            leave=False,
        ):
            pool = Pool(os.cpu_count())
            results.extend(pool.starmap(func, args_list[i : i + chunk_size]))
    else:
        raise ValueError("parallel_mtd should be none, thread, or process")
    # time_cost = timer.toc(name)
    # logger.info(f"Time cost for {name}: {time_cost:.2f} ms", verbose=False)
    # original_num = len(results)
    results = [x for x in results if x is not None]
    # logger.info(
    #     "num of args: {}, num of original results: {}, num of final results: {}".format(
    #         len(args_list), original_num, len(results), verbose=False
    #     )
    # )
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a ROS bag from KITTI odometry dataset")
    parser.add_argument("--kitti_path", default="~/data/kitti/odometry/dataset/sequences/", help="Path to the KITTI dataset")

    args = parser.parse_args()

    # for seq in range(11):
    #     create_bag(args.kitti_path, seq)
    args.kitti_path = os.path.expanduser(args.kitti_path)
    args = [(args.kitti_path, seq) for seq in range(11)]
    mt_wrapper(create_bag, args, parallel_mtd="process")