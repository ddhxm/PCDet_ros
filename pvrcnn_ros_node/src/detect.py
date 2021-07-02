
#!usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import torch
import time
from pathlib import Path

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
#from ros_marker import make_marker
from lidar_objects_msgs.msg import Object, ObjectArray
from dynamic_reconfigure.server import Server
from pvrcnn_ros_node.cfg import ThresholdsConfig
from pyquaternion import Quaternion
from geometry_msgs.msg import Point

from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from kitti_utils import Calibration
import glob

FRAME_ID = 'map'
#DETECTION_COLOR_DICT = {'Car':(255, 255, 0), 'Pedestrian':(0, 226, 255), 'Cyclist':(141, 40, 255)}
DETECTION_COLOR_DICT = [(255, 255, 0), (0, 226, 255), (141, 40, 255),(0, 255, 0), (0, 12, 0), (32, 40, 0)]
LIFETIME = 0.1

LINES = [[0, 1], [1, 2], [2, 3], [3, 0]]
LINES += [[4, 5], [5, 6], [6, 7], [7, 4]]
LINES += [[4, 0], [5, 1], [6, 2], [7, 3]]
LINES += [[4, 1], [5, 0]]

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0, 0, 1], radians=yaw)

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    return points

def remove_close(points, radius):
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points

class Processor_ROS:
    # config_path = rospy.get_param('~config_path')
    # model_path = rospy.get_param('~model_path')
    
        
    def __init__(self, config_path, model_path):
        
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/ddh/ddh/data/kitti/kitti2bag/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

        # ROS stuff
        self.sub = rospy.Subscriber("kitti_point_cloud", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        #self.sub = rospy.Subscriber("pointcloud", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        self.pub_arr_bbox = rospy.Publisher("objects", ObjectArray, queue_size=1)
        self.c_arr = None
        self.pub_marker = rospy.Publisher("id_boxes", MarkerArray, queue_size=50)

        # Dynamic reconfigure
        self.thresholds = None
        srv = Server(ThresholdsConfig, self.param_callback)

    def remove_low_score(self, image_anno):  # remove_low_score，and return img_filtered_annotations

        def get_annotations_indices(types, thresh, label_preds, scores):  # get the objects which are same to the type and satisfied the treshhold, return annotation_indices
            indexs = []
            annotation_indices = []
            for i in range(label_preds.shape[0]):
                if label_preds[i] == types:
                    indexs.append(i)
            for index in indexs:
                if scores[index] >= thresh:
                    annotation_indices.append(index)
            return annotation_indices

        img_filtered_annotations = {}
        label_preds_ = image_anno["pred_labels"].detach().cpu().numpy()  # predcted labels
        scores_ = image_anno["pred_scores"].detach().cpu().numpy()
        # get different types indices
        car_indices = get_annotations_indices(0, self.thresholds['car'], label_preds_, scores_)
        truck_indices = get_annotations_indices(1, self.thresholds['truck'], label_preds_, scores_)
        construction_vehicle_indices = get_annotations_indices(2, self.thresholds['construction'], label_preds_, scores_)
        bus_indices = get_annotations_indices(3, self.thresholds['bus'], label_preds_, scores_)
        trailer_indices = get_annotations_indices(4, self.thresholds['trailer'], label_preds_, scores_)
        barrier_indices = get_annotations_indices(5, self.thresholds['barrier'], label_preds_, scores_)
        motorcycle_indices = get_annotations_indices(6, self.thresholds['motorcycle'], label_preds_, scores_)
        bicycle_indices = get_annotations_indices(7, self.thresholds['bicycle'], label_preds_, scores_)
        pedestrain_indices = get_annotations_indices(8, self.thresholds['pedestrian'], label_preds_, scores_)
        traffic_cone_indices = get_annotations_indices(9, self.thresholds['traffic'], label_preds_, scores_)

        for key in image_anno.keys():
            if key == 'metadata':
                continue
            img_filtered_annotations[key] = (
                image_anno[key][car_indices +
                                pedestrain_indices +
                                bicycle_indices +
                                bus_indices +
                                construction_vehicle_indices +
                                traffic_cone_indices +
                                trailer_indices +
                                barrier_indices +
                                truck_indices
                                ])

        return img_filtered_annotations

    def run_detection_tracking(self, points):  # inputs points and return pred boxes,scores,types
        print("run_detection_tracking")
        t_t = time.time()
        rospy.logdebug(f"input points shape: {points.shape}")
        rospy.loginfo(f"input points shape: {points.shape}")
        num_features = 5
        self.points = points.reshape([-1, num_features])
        #self.points[:, 4] = 0  # timestamp value

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)  # trans to array

        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            pred_dicts = self.net(data_dict)[0]  
        #pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        print(f" pvrcnn inference cost time: {time.time() - t}")

        pred = self.remove_low_score(pred_dicts[0])
        boxes_lidar = pred["pred_boxes"].detach().cpu().numpy()
        #boxes_lidar[:, -1] = boxes_lidar[:, -1] + np.pi / 2
        scores = pred["pred_scores"].detach().cpu().numpy()
        types = pred["pred_labels"].detach().cpu().numpy()

        return scores, boxes_lidar, types

    def lidar_callback(self, msg):  # 
        t_t = time.time()
        arr_bbox = ObjectArray()

        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        np_p = get_xyz_points(msg_cloud, True)
        scores, dt_box_lidar, types = self.run_detection_tracking(np_p)

        arr_bbox.header.frame_id = msg.header.frame_id  # map inference coordinate:map
        arr_bbox.header.stamp = msg.header.stamp
        if scores.size != 0:
            for i in range(scores.size):
                bbox = Object()  # self-defined msg
                #q = yaw2quaternion(float(dt_box_lidar[i][6]))  # siyuanshu
                #bbox.pose.orientation.x = q[1]
                #bbox.pose.orientation.y = q[2]
                #bbox.pose.orientation.z = q[3]
                #bbox.pose.orientation.w = q[0]

                bbox.bbox3d.append(float(dt_box_lidar[i][0]))
                bbox.bbox3d.append(float(dt_box_lidar[i][1]))
                bbox.bbox3d.append(float(dt_box_lidar[i][2]))
                bbox.bbox3d.append(float(dt_box_lidar[i][3]))
                bbox.bbox3d.append(float(dt_box_lidar[i][4]))
                bbox.bbox3d.append(float(dt_box_lidar[i][5]))
                bbox.bbox3d.append(float(dt_box_lidar[i][6]))

                bbox.other_info.append(i)
                bbox.other_info.append(int(types[i]))
                bbox.other_info.append(scores[i])

                """bbox.bbox3d.x = float(dt_box_lidar[i][0])  # x
                bbox.bbox3d.y = float(dt_box_lidar[i][1])  # y
                bbox.bbox3d.z = float(dt_box_lidar[i][2])  # z
                bbox.bbox3d.h = float(dt_box_lidar[i][3])  # w
                bbox.bbox3d.w = float(dt_box_lidar[i][4])  # l
                bbox.bbox3d.l = float(dt_box_lidar[i][5])  # h
                bbox.bbox3d.ry = float(dt_box_lidar[i][6])  # h
                bbox.other_info.id = i
                bbox.other_info.label = int(types[i])
                bbox.other_info.score = scores[i]"""

                arr_bbox.bboxes.append(bbox)
        rospy.logdebug("total callback time: ", time.time() - t_t)
        print(f" total lidar_callback time:: {time.time() - t_t}")
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        if len(arr_bbox.bboxes) != 0:
            self.pub_arr_bbox.publish(arr_bbox)
            self.publish_markers(arr_bbox)
            arr_bbox.bboxes = []

    def publish_markers(self, objects_arr):  # arr_bbox.bboxes[bbox3d,other_info]
        if self.c_arr is None:
            # clear old markers
            self.c_arr = MarkerArray()
            c_m = Marker()
            c_m.header = objects_arr.header
            c_m.ns = "objects"
            c_m.id = 0
            c_m.action = Marker.DELETEALL  # delete old markers
            c_m.lifetime = rospy.Duration()
            self.c_arr.markers.append(c_m)
            c_m.ns = "ids"
            self.c_arr.markers.append(c_m)
            c_m.ns = "directions"
            self.c_arr.markers.append(c_m)
        self.pub_marker.publish(self.c_arr)

        """m_arr = MarkerArray()
        for obj in objects.bboxes:
            m_o, m_t, m_d = make_marker(obj, objects.header)
            m_arr.markers.append(m_o)
            m_arr.markers.append(m_t)
            m_arr.markers.append(m_d)
        self.pub_marker.publish(m_arr)"""


        marker_array = MarkerArray()
        bboxes_num = len(objects_arr.bboxes)
        corners_3d_velos = []
        for i in range(bboxes_num):
            bbox = objects_arr.bboxes[i].bbox3d
            corners_3d_velo = boxes_to_corners_3d(torch.unsqueeze(torch.Tensor(bbox), 0))
            print(corners_3d_velo)

            corners_3d_velos += [torch.squeeze(corners_3d_velo,0).numpy()]

        for i, corners_3d_velo in enumerate(corners_3d_velos):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = FRAME_ID

            marker.id = i
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(LIFETIME)
            marker.type = Marker.LINE_LIST

            b, g, r = DETECTION_COLOR_DICT[objects_arr.bboxes[i].other_info[1]]
            marker.color.r = r / 255.0
            marker.color.g = g / 255.0
            marker.color.b = b / 255.0
            marker.color.a = 1.0
            marker.scale.x = 0.1

            marker.points = []
            for l in LINES:
                p1 = corners_3d_velo[l[0]]
                marker.points.append(Point(p1[0], p1[1], p1[2]))
                p2 = corners_3d_velo[l[1]]
                marker.points.append(Point(p2[0], p2[1], p2[2]))
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.stamp = rospy.Time.now()
            text_marker.header.frame_id = FRAME_ID

            text_marker.id = i + 1000
            text_marker.action = Marker.ADD
            text_marker.lifetime = rospy.Duration(LIFETIME)
            text_marker.type = Marker.TEXT_VIEW_FACING

            # p4 = corners_3d_velo[4] up_left
            p = np.mean(corners_3d_velo, axis=0)  # center
            text_marker.pose.position.x = p[0]
            text_marker.pose.position.y = p[1]
            text_marker.pose.position.z = p[2] + 1

            text_marker.scale.x = 1
            text_marker.scale.y = 1
            text_marker.scale.z = 1

            b, g, r = DETECTION_COLOR_DICT[objects_arr.bboxes[i].other_info[1]]
            text_marker.color.r = r / 255.0
            text_marker.color.g = g / 255.0
            text_marker.color.b = b / 255.0
            text_marker.color.a = 1.0
            marker_array.markers.append(text_marker)
        self.pub_marker.publish(marker_array)


    def param_callback(self, config, level):
        self.thresholds = {k: v for k, v in config.items() if k != 'groups'}
        rospy.loginfo("Current thresholds: \n%s", self.thresholds)
        return config

def main():

    # pvrcnn
    rospy.init_node('pvrcnn_ros_node')
    proc = Processor_ROS(config_path, model_path)
    proc.initialize()
    rospy.loginfo("[+] pvrcnn ros_node has started!")
    rospy.spin()

if __name__ == "__main__":
    # 11~13FPS
    config_path = "/home/ddh/catkin_ws_fixed/src/PCDet_Infernece/pvrcnn_ros_node/cfgs/kitti_models/pv_rcnn.yaml"
    model_path = "/home/ddh/catkin_ws_fixed/src/PCDet_Infernece/pvrcnn_ros_node/models/pv_rcnn_8369.pth"
    calib = Calibration('/home/ddh/ddh/data/kitti/kitti2bag/2011_09_26/', from_video=True)

    main()
