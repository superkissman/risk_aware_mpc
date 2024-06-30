#include "ros/ros.h"
#include "EKF/EKF.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <unordered_map>


class Node {
private:
    ros::NodeHandle nh_;
    ros::Subscriber marker_sub_,ego_pos_sub_;
    ros::Timer timer_;
    std::unordered_map<int, EKF_CV*> ekf_map_;
    Eigen::Vector2d ego_pos_;
    

public: 
    

    Node(ros::NodeHandle& nh): nh_(nh){  
        ego_pos_ << 0, 0;   
        marker_sub_ = nh_.subscribe<visualization_msgs::MarkerArray>("/mot_tracking/box",1,&Node::markerCallback,this);
        ego_pos_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom",1,&Node::egoPosCallback,this);
        timer_ = nh_.createTimer(ros::Duration(0.1), &Node::timerCallback, this);
    }
    ~Node(){}; 

    void egoPosCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        ego_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y;
    }

    void timerCallback(const ros::TimerEvent& event) {
        for (auto& ekf : ekf_map_) {
            ekf.second->predict();
        }
    }

    void markerCallback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
        for (const auto& marker : msg->markers) {
            int id = marker.id;
            Eigen::Vector2d z;
            z << marker.pose.position.x, marker.pose.position.y;

            if (ekf_map_.find(id) == ekf_map_.end()) {
                Eigen::VectorXd x0(6);
                x0 << z, 0, 0, 0, 0;
                Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(6, 6);
                Eigen::MatrixXd Q0 = Eigen::MatrixXd::Identity(6, 6) * 0.1;
                Eigen::MatrixXd R0 = Eigen::MatrixXd::Identity(2, 2) * 0.1;
                double dt0 = 0.1;

                ekf_map_.insert(std::pair<int, EKF_CV*>(id,new EKF_CV()));
                ekf_map_[id]->initialize(x0, P0, Q0, R0, dt0);
            }

            ekf_map_[id]->predict();
            ekf_map_[id]->update(z);

            Eigen::VectorXd state = ekf_map_[id]->getState();
            ROS_INFO("Obstacle ID: %d, Position: [%f, %f]", id, state[0], state[1]);
        }
    }
       
};






int main(int argc, char  *argv[]) {
    ros::init(argc, argv, "obs_EKF");
    ros::NodeHandle nh;
    Node Node_(nh);

    ros::spin();


    return 0;
}
