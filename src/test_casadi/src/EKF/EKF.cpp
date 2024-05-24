#include "EKF.h"

// 默认构造函数
EKF::EKF() : is_initialized(false) {}

// 析构函数
EKF::~EKF() {}

// 初始化状态向量和状态协方差矩阵
void EKF::initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) {
    x = x0;
    P = P0;
    is_initialized = true;
}

// 预测步骤
void EKF::predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &Q, double dt) {
    if (!is_initialized) {
        throw std::runtime_error("EKF is not initialized!");
    }

    // 预测状态向量
    x = f(x, u, dt);

    // 计算雅可比矩阵
    Eigen::MatrixXd Fk = F(x, u, dt);

    // 更新状态协方差矩阵
    P = Fk * P * Fk.transpose() + Q;
}

// 更新步骤
void EKF::update(const Eigen::VectorXd &z, const Eigen::MatrixXd &R) {
    if (!is_initialized) {
        throw std::runtime_error("EKF is not initialized!");
    }

    // 计算测量预测值
    Eigen::VectorXd z_pred = h(x);

    // 计算测量预测雅可比矩阵
    Eigen::MatrixXd Hk = H(x);

    // 计算卡尔曼增益
    Eigen::MatrixXd S = Hk * P * Hk.transpose() + R;
    Eigen::MatrixXd K = P * Hk.transpose() * S.inverse();

    // 更新状态向量
    x = x + K * (z - z_pred);

    // 更新状态协方差矩阵
    int size = x.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
    P = (I - K * Hk) * P;
}

// 获取当前状态向量
Eigen::VectorXd EKF::getState() const {
    return x;
}

// 获取当前状态协方差矩阵
Eigen::MatrixXd EKF::getCovariance() const {
    return P;
}
