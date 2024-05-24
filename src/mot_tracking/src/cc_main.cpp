#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <time.h>
#include <unordered_map>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <deque>
#include <unordered_map>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Boost
#include <boost/tokenizer.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "readparam.h"
#include "tracking/tracker.h"
#include "detecting/detector.h"
#include "gridmapping/globalgridmap.hpp"
#include "gridmapping/localgridmap.hpp"
#include "utility.h"

// Ros
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "nav_msgs/OccupancyGrid.h" //TODO gridmap另外封装

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;
typedef pcl::PointXYZI PointType;

ros::Publisher publidar;
ros::Publisher pubmarker;
ros::Publisher pubtextmarker;
ros::Publisher pubbackground;
ros::Publisher pubobject;
ros::Publisher pubcluster;
ros::Publisher pubtracked;
ros::Publisher pubstaticobjlocal;
ros::Publisher pubdynamicobjlocal;
ros::Publisher pubstaticobjglobal;
ros::Publisher pubdynamicobjglobal;
ros::Publisher pubglobalmap;
ros::Publisher publocalmap;

ros::Subscriber subcloud;
ros::Subscriber subodom;

std::deque<nav_msgs::Odometry> odomQueue;
std::deque<sensor_msgs::PointCloud2> cloudQueue;

ros::Time timeLaserInfoStamp;

Param param;

// float z_min = -0.6, z_max = 1.5;
// float scan_range = 25;

float time_pre = 0;
double init_time;
bool init = false;

unordered_map<int, vector<int>> idcolor;
cv::RNG rng(12345);

void odomHandler(const nav_msgs::OdometryConstPtr &odomMsg)
{
	odomQueue.push_back(*odomMsg);
}

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
	cloudQueue.push_back(*cloudMsg);
	sensor_msgs::PointCloud2 lasercloudtemp = cloudQueue.front();

	pcl::PointCloud<pcl::PointXYZI>::Ptr scanPoints(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::fromROSMsg(lasercloudtemp, *scanPoints); // 激光点  intensity里是ring
	timeLaserInfoStamp = lasercloudtemp.header.stamp;

	nav_msgs::Odometry laserOdometry;
	laserOdometry = odomQueue.front(); // 里程计

	if (abs(timeLaserInfoStamp.toSec() - (laserOdometry.header.stamp.toSec()) > 0.1)) // 时间对齐
	{
		if (timeLaserInfoStamp.toSec() > laserOdometry.header.stamp.toSec())
			odomQueue.pop_front();
		else
			cloudQueue.pop_front();
		return;
	}


	cloudQueue.pop_front();
	odomQueue.pop_front();

	float relative_time;
	if (!init)
	{
		init_time = timeLaserInfoStamp.toSec();
		relative_time = 0;
		init = true;
	}
	else
	{
		relative_time = static_cast<float>(timeLaserInfoStamp.toSec() - init_time);
	}

	// 坐标转换
	double x, y, z, roll, pitch, yaw;
	x = laserOdometry.pose.pose.position.x;
	y = laserOdometry.pose.pose.position.y;
	z = laserOdometry.pose.pose.position.z;
	tf::Quaternion orientation;
	tf::quaternionMsgToTF(laserOdometry.pose.pose.orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
	Eigen::Vector2f trans = Eigen::Vector2f(x, y);
	Eigen::Matrix2f rot;
	rot << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);

	pcl::PointCloud<pcl::PointXYZI>::Ptr pointsInRange(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr pointsInRange_trans(new pcl::PointCloud<pcl::PointXYZI>());
	for (int i = 0; i < scanPoints->size(); ++i)
	{
		if (scanPoints->points[i].z > param.z_max || scanPoints->points[i].z < param.z_min)
			continue;
		// if (scanPoints->points[i].z < z_min) // 高度超过阈值的点不要
		// 	continue;

		float range = sqrt(scanPoints->points[i].x * scanPoints->points[i].x + scanPoints->points[i].y * scanPoints->points[i].y);
		if (range > param.scan_range || range < 1.5)
			continue;

		pointsInRange->push_back(scanPoints->points[i]);
	}

	// detect
	// int min_pt = 10;
	// int max_pt = 1000;
	// std::vector<float> min_size = {0.15, 0.1, 0.4}; // length,width,height
	// std::vector<float> max_size = {1.8, 1.5, 1.8};
	// int Horizon_SCAN = 1024;
	// int N_SCAN = 64;
	// int downsampleRate = 2;

	Detector detector(param.min_pt, param.max_pt, param.min_size, param.max_size, param.Horizon_SCAN, param.N_SCAN, param.downsampleRate);
	detector.setTransAndRot(trans, rot);
	std::vector<Detect> dets = detector.detect(pointsInRange); // det内的位置和点云经过坐标转换

	// track
	static Tracker tracker(param);
	std::vector<Eigen::VectorXd> result;
	std::vector<Eigen::VectorXd> tracked;
	std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> result_cloud;
	tracker.track(dets, relative_time, result, result_cloud);

	pcl::PointCloud<pcl::PointXYZI>::Ptr tracked_cloud(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr static_obj_global(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr dynamic_obj_global(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr static_obj_local(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr dynamic_obj_local(new pcl::PointCloud<pcl::PointXYZI>());

	for (int i = 0; i < result.size(); ++i)
	{
		Eigen::VectorXd r = result[i];
		for (int j = 0; j < dets.size(); ++j)
		{
			if (abs(r(1) - dets[j].position[0]) < 0.5 && abs(r(2) - dets[j].position[1]) < 0.5)
			{
				// cout << dets[j].position[0] << " " << dets[j].position[1] << endl;
				// cout << r(1) << " " << r(2)<< " " << r(3) << endl;

				pcl::PointCloud<pcl::PointXYZI>::Ptr object_local(new pcl::PointCloud<pcl::PointXYZI>());
				pcl::copyPointCloud(*pointsInRange, detector.getObjectIndex()[j], *object_local);

				// *tracked_cloud += *result_cloud[i];						   // map系下
				*tracked_cloud += *detector.getobjectVec()[j];
				tracked.push_back(r);

				if (r(3) < 0.55) // 速度小于阈值
				{
					*static_obj_global += *detector.getobjectVec()[j];
					*static_obj_local += *object_local;
				}
				else
				{
					*dynamic_obj_global += *detector.getobjectVec()[j];
					*dynamic_obj_local += *object_local;
					// std::ofstream foutC("/home/gaoe/Mount/MOT/tracking.txt", std::ios::app);
					// foutC.setf(std::ios::fixed, std::ios::floatfield);
					// foutC.precision(5);
					// foutC << r(0) << " " << r(1) << " " << r(2) << " " << r(3) << " " << r(4) << " " << r(5) << endl;
					// foutC.close();
				}
				break;
			}
		}
	}

	// cout << "*****" << endl;

	pcl::VoxelGrid<PointType> downSizeFilter;
	downSizeFilter.setLeafSize(0.3, 0.3, 0.3);

	pcl::PointCloud<pcl::PointXYZI>::Ptr bgcloud_local(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::PointCloud<pcl::PointXYZI>::Ptr bgcloud_localDS(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::copyPointCloud(*pointsInRange, detector.getBackgroundIndex(), *bgcloud_local);
	downSizeFilter.setInputCloud(bgcloud_local);
	downSizeFilter.filter(*bgcloud_localDS);
	int localmapsize=param.scan_range*2;
	// float localresolution=0.2;
	LocalMap localmap(localmapsize, param.localresolution, bgcloud_localDS, static_obj_local,dynamic_obj_local);
	localmap.process();
	Eigen::MatrixXf localmap_probability = localmap.getMapbability();
	Eigen::MatrixXi localmap_state = localmap.getMapstate();

	int rowslocal = localmap_probability.rows();
	int colslocal  = localmap_probability.cols();
	nav_msgs::OccupancyGrid localgridmap;
	localgridmap.header.stamp = lasercloudtemp.header.stamp;
	localgridmap.header.frame_id = lasercloudtemp.header.frame_id;
	localgridmap.info.resolution = param.localresolution;
	localgridmap.info.width = rowslocal ;
	localgridmap.info.height = colslocal ;
	localgridmap.info.origin.position.x = -floor(localmapsize / 2);
	localgridmap.info.origin.position.y = -floor(localmapsize / 2);
	localgridmap.info.origin.position.z = 0;
	localgridmap.info.origin.orientation.w = 1;
	localgridmap.info.origin.orientation.x = 0;
	localgridmap.info.origin.orientation.y = 0;
	localgridmap.info.origin.orientation.z = 0;
	const int Nlocal  = rowslocal  * colslocal ;
	for (int i = 0; i < Nlocal ; ++i)
	{
		float value = localmap_probability(i);
		int state=localmap_state(i);
		if (value == 0.5)
			localgridmap.data.push_back(-1); // 未知
		else if (value > 0.6)
		{
			if(state==1)
			localgridmap.data.push_back(100); // 环境
			else if(state==2)
			localgridmap.data.push_back(127); // 静态
			else if(state==3)
			localgridmap.data.push_back(-127); // 静态
		}
		else
			localgridmap.data.push_back(0); // 空闲
	}
	publocalmap.publish(localgridmap);

	pcl::PointCloud<pcl::PointXYZI>::Ptr objcloud_global(new pcl::PointCloud<pcl::PointXYZI>(detector.getobject()));
	pcl::PointCloud<pcl::PointXYZI>::Ptr bgcloud_global(new pcl::PointCloud<pcl::PointXYZI>(detector.getbackground())); // map系下
	pcl::PointCloud<pcl::PointXYZI>::Ptr bgcloud_globalDS(new pcl::PointCloud<pcl::PointXYZI>());						// map系下

	downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
	downSizeFilter.setInputCloud(bgcloud_global);
	downSizeFilter.filter(*bgcloud_globalDS);
	// int globalmapsize = 300;
	// float globalresolution = 0.2;
	GlobalMap globalmap(param.globalmapsize, param.globalresolution, bgcloud_globalDS, trans);
	globalmap.process();
	Eigen::MatrixXf globalmap_probability = globalmap.getMapbability();
	int rows = globalmap_probability.rows();
	int cols = globalmap_probability.cols();
	nav_msgs::OccupancyGrid gridmap;
	gridmap.header.stamp = lasercloudtemp.header.stamp;
	gridmap.header.frame_id = "map";
	gridmap.info.resolution = param.globalresolution;
	gridmap.info.width = rows;
	gridmap.info.height = cols;
	gridmap.info.origin.position.x = -floor(param.globalmapsize / 2);
	gridmap.info.origin.position.y = -floor(param.globalmapsize / 2);
	gridmap.info.origin.position.z = 0;
	gridmap.info.origin.orientation.w = 1;
	gridmap.info.origin.orientation.x = 0;
	gridmap.info.origin.orientation.y = 0;
	gridmap.info.origin.orientation.z = 0;
	const int N = rows * cols;
	for (int i = 0; i < N; ++i)
	{
		float value = globalmap_probability(i);
		if (value == 0.5)
			gridmap.data.push_back(-1); // 未知
		else if (value > 0.6)
		{
			gridmap.data.push_back(100); // 占据
		}
		else
			gridmap.data.push_back(0); // 空闲
	}
	pubglobalmap.publish(gridmap);

	// publish marker
	visualization_msgs::MarkerArrayPtr marker_array(new visualization_msgs::MarkerArray());
	marker_array->markers.reserve(tracked.size() + 1);
	int marker_id = 0;
	for (auto &r : tracked)
	{
		if (!idcolor.count(int(r(0)))) // r(0)=id
		{
			// 随机生成一个颜色
			int red = rng.uniform(0, 255);
			int green = rng.uniform(0, 255);
			int blue = rng.uniform(0, 255);
			idcolor[int(r(0))] = {red, green, blue};
		}
		visualization_msgs::Marker marker;
		marker.header.stamp = lasercloudtemp.header.stamp;
		marker.header.frame_id = "map";
		marker.ns = "";
		marker.id = marker_id;
		marker.lifetime = ros::Duration(1.0);
		marker.type = visualization_msgs::Marker::CUBE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = r(1);
		marker.pose.position.y = r(2);
		marker.pose.position.z = 0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = r(6);
		marker.scale.y = r(7);
		marker.scale.z = r(8);
		marker.color.r = float(idcolor[int(r(0))][0]) / 255;
		marker.color.g = float(idcolor[int(r(0))][1]) / 255;
		marker.color.b = float(idcolor[int(r(0))][2]) / 255;
		marker.color.a = 0.6f;
		marker_array->markers.push_back(marker); 

		++marker_id;
	}
	pubmarker.publish(marker_array);

	// publish cloud
	vector<vector<int>> clustersIndex = detector.getClustersIndex();
	for (auto cluster_vec : clustersIndex)
	{
		int intensity = rand() % 255;
		for (int i = 0; i < cluster_vec.size(); ++i)
		{
			pointsInRange->points[cluster_vec[i]].intensity = intensity;
		}
	}
	auto cluster_indices = detector.getMergedClustersIndex();
	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::copyPointCloud(*pointsInRange, cluster_indices, *cluster_points);
	sensor_msgs::PointCloud2 laserCloudTemp;
	pcl::toROSMsg(*cluster_points, laserCloudTemp);
	laserCloudTemp.header.stamp = lasercloudtemp.header.stamp;
	laserCloudTemp.header.frame_id = lasercloudtemp.header.frame_id;
	pubcluster.publish(laserCloudTemp);

	sensor_msgs::PointCloud2 obj;
	pcl::toROSMsg(*objcloud_global, obj);
	obj.header.stamp = lasercloudtemp.header.stamp;
	obj.header.frame_id = "map";
	pubobject.publish(obj);

	sensor_msgs::PointCloud2 bg;
	pcl::toROSMsg(*bgcloud_global, bg);
	bg.header.stamp = lasercloudtemp.header.stamp;
	bg.header.frame_id = "map";
	pubbackground.publish(bg);

	sensor_msgs::PointCloud2 trackedcloud;
	pcl::toROSMsg(*tracked_cloud, trackedcloud);
	trackedcloud.header.stamp = lasercloudtemp.header.stamp;
	trackedcloud.header.frame_id = "map";
	pubtracked.publish(trackedcloud);

	sensor_msgs::PointCloud2 staticcloudlocal;
	pcl::toROSMsg(*static_obj_local, staticcloudlocal);
	staticcloudlocal.header = lasercloudtemp.header;
	pubstaticobjlocal.publish(staticcloudlocal);

	sensor_msgs::PointCloud2 dynamiccloudlocal;
	pcl::toROSMsg(*dynamic_obj_local, dynamiccloudlocal);
	dynamiccloudlocal.header = lasercloudtemp.header;
	pubdynamicobjlocal.publish(dynamiccloudlocal);

	sensor_msgs::PointCloud2 staticcloudglobal;
	pcl::toROSMsg(*static_obj_global, staticcloudglobal);
	staticcloudglobal.header.stamp = lasercloudtemp.header.stamp;
	staticcloudglobal.header.frame_id = "map";
	pubstaticobjglobal.publish(staticcloudglobal);

	sensor_msgs::PointCloud2 dynmaiccloudglobal;
	pcl::toROSMsg(*dynamic_obj_global, dynmaiccloudglobal);
	dynmaiccloudglobal.header.stamp = lasercloudtemp.header.stamp;
	dynmaiccloudglobal.header.frame_id = "map";
	pubdynamicobjglobal.publish(dynmaiccloudglobal);
}

int main(int argc, char **argv)
{

	ros::init(argc, argv, "tracking_mapping_node");

	ROS_INFO("\033[1;32m----> Tracking and Mapping Started.\033[0m");


	ros::NodeHandle nh;

	image_transport::ImageTransport it(nh);
	publidar = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/pointcloud", 1);
	pubmarker = nh.advertise<visualization_msgs::MarkerArray>("/mot_tracking/box", 1);
	// pubtextmarker = nh.advertise<visualization_msgs::MarkerArray>("/mot_tracking/id", 1);

	pubbackground = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/background_global", 1);
	pubobject = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/object_global", 1);
	pubcluster = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/clusters_local", 1);
	pubtracked = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/tracked_global", 1);
	pubstaticobjlocal = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/static_obj_local", 1);
	pubdynamicobjlocal = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/dynamic_obj_local", 1);
	pubstaticobjglobal = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/static_obj_global", 1);
	pubdynamicobjglobal = nh.advertise<sensor_msgs::PointCloud2>("/mot_tracking/dynamic_obj_global", 1);
	pubglobalmap = nh.advertise<nav_msgs::OccupancyGrid>("/global_map", 1);
	publocalmap = nh.advertise<nav_msgs::OccupancyGrid>("/local_map", 1);

	subcloud = nh.subscribe<sensor_msgs::PointCloud2>("lio_plane2plane/mapping/cloudDS", 5, cloudHandler);
	subodom = nh.subscribe<nav_msgs::Odometry>("lio_plane2plane/mapping/odometry", 5, odomHandler);

	ros::spin();

	return 0;
}
