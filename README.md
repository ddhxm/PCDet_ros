# PCDet_Infernece
此节点主要用于实现基于激光雷达的目标检测，此框架是基于OpenPCDet封装到ros上。
硬件环境：NVIDIA GTX3070，pandar64 
软件环境：Ubuntu18.04，ros-melodic，python3.6，pytorch1.8.1+cu111

OpenPCDet的安装参考其INSTALL.md：参照home目录下的OpenPCDet
1.  终端输入git clone --recursive https://github.com/open-mmlab/OpenPCDet.git
2.  pip3 install -r requirements.txt,此处注意其内部要求的torch版本，需与本机的cuda类型相对应，例如，本项目使用NVIDIA GTX3070，由于其驱动等限制，CUDA只能使用11以上的版本，安装的torch版本与cuda版本不匹配，会出现无法使用GPU加速的情况，因此，请大家确定好环境版本，再进行安装。若直接通过命令行pip3 requirements.txt建立的torch有可能与cuda不匹配，因此，建议先行安装cuda对应版本的torch，并将requirements.txt中的torch版本要求去除，再进行安装。
3.  按照要求编译spconv包，参考此链接下版本要求：cmake >= 3.13.2
git clone --recursive https://github.com/traveller59/spconv
若torch>=1.0，则输入python setup.py bdist_wheel
cd ./dist, 并pip XXX.whl进行安装spconv
4. sudo python3 setup.py develop安装OpenPCDet
注： 在编译过程中可能遇到llvm、llvmlite和numba安装问题，安装numba要求llvm>9.0,若环境中llvm达不到要求，直接下载编译好llvm的ubuntu的库，命令行增加LLVM_CONFIG=安装路径，并pip3 install llvmlite==0.36.0
pip3 install numba报错tbb，版本过低，GitHub安装最新版本，编译通过，安装成功。

目录梳理：
	此文件夹下主要文件夹为lidar_objects_msgs,其主要为自定义的检测目标消息，pvrcnn_ros_node为节点相关，其目录内的src文件夹中包含detect.py文件，为kitti连续帧的检测结果，包括在rviz中的可视化，目前的激光雷达目标检测还只是基于kitti数据集进行，还未实时在现实环境中检测，对于后者，需要更改detect.py文件中的数据读入方式，此部分后续进行更新。
运行流程如下：
1.  启动激光雷达的节点：
	roslaunch hesai_lidar hesai_lidar.launch lidar_type:="Pandar64" frame_id:="Pandar64"
2.  打开新终端，输入python3 detect.py(注意为python3)
3.  rosrun rviz rviz，通过topic添加检测结果进行可视化

注：kitti数据集的发布在/home/ddh/ddh/data/kitti_visual/scripts中python kitti.py,数据集在此路径中：/home/ddh/ddh/data/kitti/kitti2bag/2011_09_26

detect.py文件解读：
此程序通过订阅话题kitti_point_cloud（上述注中的发布话题名称）得到点云数据（后期若接入实际点云数据，通过改变其对应的话题名称即可），并发布话题名为objects的检测框，其数据形式为自定义消息lidar_objects_msgs中的ObjectArray，包含N个消息类型为Object的检测框，其同在lidar_objects_msgs中进行定义；同时发布话题id_boxes用于rviz中的显示和可视化。


