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

def create_bag(kitti_path):
    bridge = CvBridge()

    image_dir = os.path.join(kitti_path, '00', 'image_2')
    velodyne_dir = os.path.join(kitti_path, '00', 'velodyne')
    timestamps_path = os.path.join(kitti_path, '00', 'times.txt')

    with open(timestamps_path, 'r') as f:
        timestamps = f.readlines()

    bag_path = os.path.join(kitti_path, '00', 'data.bag')
    bag = rosbag.Bag(bag_path, 'w')

    # write velodyne
    for idx, timestamp in enumerate(tqdm(timestamps)):
        timestamp = float(timestamp.strip())
        timestamp = rospy.Time.from_sec(timestamp)

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a ROS bag from KITTI odometry dataset")
    parser.add_argument("--kitti_path", default="/home/qzj/data/KITTI/odometry/dataset/sequences/", help="Path to the KITTI dataset")

    args = parser.parse_args()

    create_bag(args.kitti_path)