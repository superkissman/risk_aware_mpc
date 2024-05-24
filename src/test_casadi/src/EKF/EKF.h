#ifndef _EKF_H
#define _EKF_H

#include <Eigen/Dense>

class EKF {
public:
    EKF();
    ~EKF();

    void initialize(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);
    void predict(const Eigen::VectorXd &u, const Eigen::MatrixXd &Q, double dt);
    void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &R);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

protected:
    virtual Eigen::VectorXd f(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt) = 0;
    virtual Eigen::MatrixXd F(const Eigen::VectorXd &x, const Eigen::VectorXd &u, double dt) = 0;
    virtual Eigen::MatrixXd H(const Eigen::VectorXd &x) = 0;
    virtual Eigen::VectorXd h(const Eigen::VectorXd &x) = 0;

private:
    Eigen::VectorXd x; // 状态向量
    Eigen::MatrixXd P; // 状态协方差矩阵
    bool is_initialized;
};

#endif // EKF_H
