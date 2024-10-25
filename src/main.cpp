#include "ceres/ceres.h"
#include "glog/logging.h"

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
    template < typename T >
	bool operator() (const T * const x, T * residual) const {
	residual[0] = 10.0 - x[0];
	return true;
}};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    double x = 0.5;
    const double initial_x = x;

    // Build the problem.
    ceres::Problem problem;

    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    //   <CostFunctor, 1, 1>
    // CostFunctor: 定义的残差函数类型
    // 1: 残差的维度
    // 1: 优化变量的维度
    ceres::CostFunction * cost_function =
	new ceres::AutoDiffCostFunction < CostFunctor, 1,
	1 > (new CostFunctor);

    // Ceres 中用来向优化问题添加残差块的方法
    // cost_function: 残差函数
    // nullptr: 损失函数 (不使用任何损失函数，即将误差的平方直接用于优化)
    // &x: 指向我们要优化的参数 x
    problem.AddResidualBlock(cost_function, nullptr, &x);

    // Run the solver!
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x << " -> " << x << "\n";
    return 0;
}
