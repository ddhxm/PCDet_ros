#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import torch
import time
from pathlib import Path

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from ros_marker import make_marker
from lidar_objects_msgs.msg import Object, ObjectArray
from dynamic_reconfigure.server import Server
from pvrcnn_ros_node.cfg import ThresholdsConfig
from pyquaternion import Quaternion

from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils


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
            root_path=Path("/home/cds-josh/bin_2019_bag/000001.bin"),
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
        t_t = time.time()
        rospy.logdebug(f"input points shape: {points.shape}")
        num_features = 5
        self.points = points.reshape([-1, num_features])
        #self.points[:, 4] = 0  # timestamp value

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            pred_dicts = self.net(data_dict)[0]  
        #pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        print(f" pvrcnn inference cost time: {time.time() - t}")

        pred = self.remove_low_score(pred_dicts[0])
        #print(pred.keys())
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

        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        if scores.size != 0:
            for i in range(scores.size):
                bbox = Object()  # self-defined msg
                q = yaw2quaternion(float(dt_box_lidar[i][6]))  # siyuanshu
                bbox.pose.orientation.x = q[1]
                bbox.pose.orientation.y = q[2]
                bbox.pose.orientation.z = q[3]
                bbox.pose.orientation.w = q[0]
                bbox.pose.position.x = float(dt_box_lidar[i][0])  # x
                bbox.pose.position.y = float(dt_box_lidar[i][1])  # y
                bbox.pose.position.z = float(dt_box_lidar[i][2])  # z
                bbox.size.x = float(dt_box_lidar[i][4])  # w
                bbox.size.y = float(dt_box_lidar[i][3])  # l
                bbox.size.z = float(dt_box_lidar[i][5])  # h
                bbox.score = scores[i]
                bbox.id = i
                bbox.label = int(types[i])
                arr_bbox.objects.append(bbox)
        rospy.logdebug("total callback time: ", time.time() - t_t)
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        if len(arr_bbox.objects) != 0:
            self.pub_arr_bbox.publish(arr_bbox)
            self.publish_markers(arr_bbox)
            arr_bbox.objects = []

    def publish_markers(self, objects):
        if self.c_arr is None:
            # clear old markers
            self.c_arr = MarkerArray()
            c_m = Marker()
            c_m.header = objects.header
            c_m.ns = "objects"
            c_m.id = 0
            c_m.action = Marker.DELETEALL
            c_m.lifetime = rospy.Duration()
            self.c_arr.markers.append(c_m)
            c_m.ns = "ids"
            self.c_arr.markers.append(c_m)
            c_m.ns = "directions"
            self.c_arr.markers.append(c_m)
        self.pub_marker.publish(self.c_arr)

        m_arr = MarkerArray()
        for obj in objects.objects:
            m_o, m_t, m_d = make_marker(obj, objects.header)
            m_arr.markers.append(m_o)
            m_arr.markers.append(m_t)
            m_arr.markers.append(m_d)
        self.pub_marker.publish(m_arr)

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
    config_path = "/home/cds-josh/OpenPCDet_ws/src/pvrcnn_ros_node/cfgs/nuscenes_models/cbgs_second_multihead.yaml"  
    model_path = "/home/cds-josh/OpenPCDet_ws/src/pvrcnn_ros_node/models/cbgs_second_multihead_nds6229_updated.pth"  

    main()
