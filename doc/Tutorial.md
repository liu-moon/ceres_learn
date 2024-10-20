# 教程

## 非线性最小二乘

### 介绍

Ceres 可以解决以下形式的非线性最小二乘问题

$$
\begin{split}\min_{\mathbf{x}} &\quad \frac{1}{2}\sum_{i} \rho_i\left(\left\|f_i\left(x_{i_1}, ... ,x_{i_k}\right)\right\|^2\right) \\
\text{s.t.} &\quad l_j \le x_j \le u_j\end{split}
\tag{1}
$$

这种形式的问题广泛出现在科学和工程领域——从统计学中的拟合曲线到计算机视觉中根据照片构建3D模型。

在本章中，我们将学习如何使用Ceres Solver解决公式$(1)$的问题，本章中描述的素有示例以及更多示例的完整代码可以在示例目录中找到。

表达式$\rho_i\left(\left\|f_i\left(x_{i_1},...,x_{i_k}\right)\right\|^2\right)$称为`ResidualBlock`，其中$f_i(\cdot)$是一个`CostFunction`依赖于参数块$\left[x_{i_1},... , x_{i_k}\right]$。在大多数优化问题中，小组标量会一起出现。例如，平移向量的三个分量和定义相机姿势的四元数的四个分量。我们将这样一组小标量称为`ParameterBlock`。当然，`ParameterBlock`可以只是一个参数。
$l_j$和$u_j$是参数块$x_j$的界限。

$\rho_i$是`LossFunction`。`LossFunction`是一个标量函数，用于减少异常值对非线性最小二乘问题解的影响。

作为特例，当$\rho_i(x) = x$，即恒等函数，且$l_j = -\infty$和$u_j = \infty$时，我们得到了更为熟悉的非线性最小二乘问题。

$$
\frac{1}{2}\sum_{i} \left\|f_i\left(x_{i_1}, ... ,x_{i_k}\right)\right\|^2.
\tag{2}
$$

### Hello World

首先，考虑寻找函数最小值的问题。

$$
\frac{1}{2}(10 -x)^2.
$$

这是一个很简单的问题，其最小值位于$x = 10$，但它是一个很好的起点，可以说明使用 Ceres 解决问题的基础知识

第一步是编写一个functor来评估这个函数$f(x) = 10 - x$

```c++
struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = 10.0 - x[0];
     return true;
   }
};
```

这里要注意的重要一点是，`operator()`是一个模板方法，它假定其所有输入和输出都属于某种类型`T`。
这里使用模板允许 Ceres 调用 `CostFunctor::operator<T>()`，当只需要残差的值时使用 `T=double`，当需要雅可比矩阵时使用特殊类型 `T=Jet`。在导数部分，我们将更详细地讨论向Ceres提供导数的各种方式。

一旦我们有了计算残差函数的方法，现在就可以使用它构建非线性最小二乘问题并让 Ceres 解决它。

```c++
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // 求解变量初值
  double initial_x = 5.0;
  double x = initial_x;

  // 构建问题
  Problem problem;

  // 设置唯一的cost_function（也称为残差）。使用自动微分来求导数（雅可比矩阵）
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>();
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // 运行求解器！
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
```

`AutoDiffCostFunction` 以 `CostFunctor` 作为输入，自动区分它并为其提供 `CostFunction` 接口。

编译并运行 `examples/helloworld.cc` 得到

```shell
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.512500e+01    0.00e+00    9.50e+00   0.00e+00   0.00e+00  1.00e+04        0    7.87e-06    2.53e-03
   1  4.511598e-07    4.51e+01    9.50e-04   0.00e+00   1.00e+00  3.00e+04        1    5.48e-05    2.61e-03
   2  5.012552e-16    4.51e-07    3.17e-08   9.50e-04   1.00e+00  9.00e+04        1    4.05e-06    2.62e-03
Ceres Solver Report: Iterations: 3, Initial cost: 4.512500e+01, Final cost: 5.012552e-16, Termination: CONVERGENCE
x : 0.5 -> 10
```

从$x=5$开始，求解器经过两次迭代达到 10。细心的读者会注意到这是一个线性问题，一次线性求解就足以获得最优值。求解器的默认配置针对的是非线性问题，为了简单起见，我们在本例中没有更改它。确实可以在一次迭代中使用 Ceres 获得此问题的解。还请注意，求解器在第一次迭代中确实非常接近最优函数值 0。当我们讨论 Ceres 的收敛和参数设置时，我们将更详细地讨论这些问题。

| iter        | 迭代次数    | 当前的迭代次数，从0开始                                                    |
|-------------|---------|-----------------------------------------------------------------|
| cost        | 代价函数值   | 目标函数的当前值，表示优化过程中计算的代价值（目标值）。Ceres Solver的目标是最小化这个值。             |
| cost_change | 代价变化量   | 当前迭代与上一次迭代相比，代价函数的变化量。这个值越小，说明优化正在收敛。                           |
| &#124; gradient &#124;  | 梯度的范数   | 目标函数的梯度（或偏导数）的范数（大小），反映了代价函数的变化率。梯度越小，说明距离最优解越近。                |
| &#124; step &#124;      | 步长的范数   | 当前迭代中优化变量的更新步长，表示每次迭代中调整的幅度。步长越小，说明调整的幅度越小，通常与收敛接近相关。           |
| tr_ratio    | 信赖域比率   | 信赖域子问题的实际代价减少和预测代价减少的比率，反映了信赖域方法的收敛情况。比率接近1说明模型的预测与实际相符，优化效果较好。 |
| tr_radius   | 信赖域半径   | 当前迭代中信赖域的半径，表示优化中信赖域的大小。信赖域半径决定了每次迭代的步长。                        |
| ls_iter     | 线搜索迭代次数 | 在每次迭代中，为找到合适步长而进行的线搜索迭代次数。                                      |
| iter_time   | 每次迭代耗时  | 每次迭代的时间，单位为秒。可以用来评估每次迭代的效率。                                     |
| total_time  | 总耗时     | 从优化过程开始到当前迭代为止的总耗时。                                             |



### 导数

Ceres Solver 与大多数优化库一样，依赖于能够评估目标函数中每个项在任意参数值下的值和导数。正确有效地执行求导操作对于获得良好结果至关重要。Ceres Solver 提供了多种方法。您已经在 `examples/helloworld.cc` 中看到了其中一种实际应用——自动微分。

我们现在考虑另外两种可能性。解析导数和数值导数。

#### 数值导数

在某些情况下，无法定义模板化cost functor，例如当残差的评估涉及调用您无法控制的库函数时。在这种情况下，可以使用数值微分。用户定义一个计算残差值的functor，并使用它构造一个 `NumericDiffCostFunction`。例如，对于$ f(x) = 10 - x$，相应的functor将是

```c++
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
```

添加到`Problem`中如下：

```c++
CostFunction* cost_function =
  new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>();
problem.AddResidualBlock(cost_function, nullptr, &x);
```

注意我们使用自动微分时的相似之处

```c++
CostFunction* cost_function =
    new AutoDiffCostFunction<CostFunctor, 1, 1>();
problem.AddResidualBlock(cost_function, nullptr, &x);
```

该构造看起来与用于自动微分的构造几乎相同，除了一个额外的模板参数，该参数指示用于计算数值导数的有限差分方案的类型。有关更多详细信息，请参阅 `NumericDiffCostFunction` 的文档

一般来说，我们建议使用自动微分而不是数值微分。使用 C++ 模板可以使自动微分更加高效，而数值微分则成本高昂、容易出现数值错误，并且会导致收敛速度变慢。

#### 解析导数

在某些情况下，使用自动微分是不可能的。例如，可能的情况是，以封闭形式计算导数而不是依赖于自动微分代码使用的链式法则更有效。在这种情况下，可以提供您自己的残差和雅可比计算代码。为此，如果您在编译时知道参数和残差的大小，请定义 `CostFunction` 或 `SizedCostFunction` 的子类。例如，这里是实现$ f(x) = 10 -
x $的 `QuadraticCostFunction`。

```c++
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  virtual ~QuadraticCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = 10 - x;

    // Compute the Jacobian if asked for.
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};
```

`QuadraticCostFunction::Evaluate`提供了参数输入数组`parameters`、残差输出数组` residuals `和雅可比矩阵输出数组` jacobians`。`jacobians`矩阵数组是可选的，`Evaluate` 会检查它是否为非空，如果是，则用残差函数导数的值填充它。在这种情况下，由于残差函数是线性的，因此雅可比矩阵是常数

从上面的代码片段可以看出，实现 `CostFunction` 对象有点繁琐。我们建议，除非您有充分的理由自己管理雅可比矩阵计算，否则请使用 `AutoDiffCostFunction` 或 `NumericDiffCostFunction` 来构建残差块。

#### 更多关于导数的信息

计算导数是使用 Ceres 最复杂的部分，根据具体情况，用户可能需要更复杂的方法来计算导数。本节只是介绍了如何向 Ceres 提供导数。一旦您熟悉了使用 `NumericDiffCostFunction` 和 `AutoDiffCostFunction`，我们建议您查看 `DynamicAutoDiffCostFunction`、`CostFunctionToFunctor`、`NumericDiffFunctor` 和 `ConditionedCostFunction`，以了解构建和计算成本函数的更高级方法。



### Powell’s Function

现在考虑一个稍微复杂一点的例子——Powell’s函数的最小化。设 $x = \left[x_1, x_2, x_3, x_4 \right]$
且

$$
\begin{split}\begin{align}
f_1(x) &= x_1 + 10x_2 \\
f_2(x) &= \sqrt{5} (x_3 - x_4)\\
f_3(x) &= (x_2 - 2x_3)^2\\
f_4(x) &= \sqrt{10} (x_1 - x_4)^2\\
F(x) &= \left[f_1(x),\ f_2(x),\ f_3(x),\ f_4(x) \right]
\end{align}\end{split}
$$

$F(x)$ 是四个参数的函数，有四个残差，我们希望找到$x$使得$\frac{1}{2}\|F(x)\|^2$被最小化。

同样，第一步是定义用于评估目标项的functor。以下是评估$ f_4(x_1, x_4)$ 的代码：

~~~c++
struct F4 {
  template <typename T>
  bool operator()(const T* const x1, const T* const x4, T* residual) const {
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};
~~~

类似地，我们可以定义类 `F1`、`F2` 和 `F3` 来分别评估$f_1(x_1, x_2)$ 、$f_2(x_3, x_4)$和 $f_3(x_2, x_3)$。利用这些，可以构建问题如下：

```c++
double x1 =  3.0; double x2 = -1.0; double x3 =  0.0; double x4 = 1.0;

Problem problem;

// Add residual terms to the problem using the autodiff
// wrapper to get the derivatives automatically.
problem.AddResidualBlock(
  new AutoDiffCostFunction<F1, 1, 1, 1>(), nullptr, &x1, &x2);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F2, 1, 1, 1>(), nullptr, &x3, &x4);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F3, 1, 1, 1>(), nullptr, &x2, &x3);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F4, 1, 1, 1>(), nullptr, &x1, &x4);
```

请注意，每个 `ResidualBlock` 仅依赖于相应残差对象所依赖的两个参数，而不是所有四个参数。编译并运行 examples/powell.cc 会得到：

```shell
Initial x1 = 3, x2 = -1, x3 = 0, x4 = 1
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  1.075000e+02    0.00e+00    1.55e+02   0.00e+00   0.00e+00  1.00e+04        0    1.60e-05    9.20e-05
   1  5.036190e+00    1.02e+02    2.00e+01   0.00e+00   9.53e-01  3.00e+04        1    6.70e-05    2.20e-04
   2  3.148168e-01    4.72e+00    2.50e+00   6.23e-01   9.37e-01  9.00e+04        1    1.60e-05    2.56e-04
   3  1.967760e-02    2.95e-01    3.13e-01   3.08e-01   9.37e-01  2.70e+05        1    1.12e-05    2.88e-04
   4  1.229900e-03    1.84e-02    3.91e-02   1.54e-01   9.37e-01  8.10e+05        1    1.10e-05    3.25e-04
   5  7.687123e-05    1.15e-03    4.89e-03   7.69e-02   9.37e-01  2.43e+06        1    9.06e-06    3.46e-04
   6  4.804625e-06    7.21e-05    6.11e-04   3.85e-02   9.37e-01  7.29e+06        1    8.82e-06    3.69e-04
   7  3.003028e-07    4.50e-06    7.64e-05   1.92e-02   9.37e-01  2.19e+07        1    9.06e-06    3.92e-04
   8  1.877006e-08    2.82e-07    9.54e-06   9.62e-03   9.37e-01  6.56e+07        1    9.06e-06    4.14e-04
   9  1.173223e-09    1.76e-08    1.19e-06   4.81e-03   9.37e-01  1.97e+08        1    8.82e-06    4.37e-04
  10  7.333425e-11    1.10e-09    1.49e-07   2.40e-03   9.37e-01  5.90e+08        1    9.06e-06    4.59e-04
  11  4.584044e-12    6.88e-11    1.86e-08   1.20e-03   9.37e-01  1.77e+09        1    8.11e-06    4.82e-04
  12  2.865573e-13    4.30e-12    2.33e-09   6.02e-04   9.37e-01  5.31e+09        1    9.06e-06    5.11e-04
  13  1.791438e-14    2.69e-13    2.91e-10   3.01e-04   9.37e-01  1.59e+10        1    8.82e-06    5.34e-04
  14  1.120029e-15    1.68e-14    3.64e-11   1.51e-04   9.37e-01  4.78e+10        1    8.11e-06    5.56e-04

Solver Summary (v 2.2.0-eigen-(3.4.0)-lapack-suitesparse-(7.6.1)-eigensparse)

                                     Original                  Reduced
Parameter blocks                            4                        4
Parameters                                  4                        4
Residual blocks                             4                        4
Residuals                                   4                        4

Minimizer                        TRUST_REGION

Dense linear algebra library            EIGEN 
Trust region strategy     LEVENBERG_MARQUARDT
                                        Given                     Used
Linear solver                        DENSE_QR                 DENSE_QR
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                        4

Cost:
Initial                          1.075000e+02
Final                            1.120029e-15
Change                           1.075000e+02

Minimizer iterations                       15
Successful steps                           15
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.000076

  Residual only evaluation           0.000014 (14)
  Jacobian & residual evaluation     0.000025 (15)
  Linear solver                      0.000077 (14)
Minimizer                            0.000501

Postprocessor                        0.000006
Total                                0.000583

Termination:                      CONVERGENCE (Gradient tolerance reached. Gradient max norm: 3.642190e-11 <= 1.000000e-10)

Final x1 = 0.000146222, x2 = -1.46222e-05, x3 = 2.40957e-05, x4 = 2.40957e-05
```

很容易看出，该问题的最优解位于$x_1=0, x_2=0, x_3=0, x_4=0$处，目标函数值为$0
$。在 10 次迭代中，Ceres 找到了一个目标函数值为 $4\times 10^{-12}$ 的解。



### 曲线拟合

到目前为止我们看到的例子都是没有数据的简单优化问题。最小二乘和非线性最小二乘分析的最初目的是将曲线拟合到数据中。现在我们来考虑这样一个问题的例子是再合适不过的了。它包含通过对曲线 $y =
e^{0.3x + 0.1}$进行采样并添加标准差为 $\sigma = 0.2$的高斯噪声生成的数据。让我们将一些数据拟合到曲线
$$
y = e^{mx + c}
$$

我们首先定义一个模板对象来评估残差。每个观测值都会有一个残差。

```c++
struct ExponentialResidual {
  ExponentialResidual(double x, double y)
      : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* const m, const T* const c, T* residual) const {
    residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

 private:
  // Observations for a sample.
  const double x_;
  const double y_;
};
```

假设观测值位于一个 $2n$大小的数组中，称为`data`，那么问题的构建很简单，只需为每个观测值创建一个 `CostFunction`。

```c++
double m = 0.0;
double c = 0.0;

Problem problem;
for (int i = 0; i < kNumObservations; ++i) {
  CostFunction* cost_function =
       new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>
           (data[2 * i], data[2 * i + 1]);
  problem.AddResidualBlock(cost_function, nullptr, &m, &c);
}
```

编译并运行 examples/curve_fitting.cc 得到以下结果：

```c++
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  1.211734e+02    0.00e+00    3.61e+02   0.00e+00   0.00e+00  1.00e+04        0    2.00e-05    5.41e-05
   1  2.334822e+03   -2.21e+03    3.61e+02   0.00e+00  -1.87e+01  5.00e+03        1    2.60e-05    1.10e-03
   2  2.331438e+03   -2.21e+03    3.61e+02   0.00e+00  -1.86e+01  1.25e+03        1    7.15e-06    1.12e-03
   3  2.311313e+03   -2.19e+03    3.61e+02   0.00e+00  -1.85e+01  1.56e+02        1    5.96e-06    1.13e-03
   4  2.137268e+03   -2.02e+03    3.61e+02   0.00e+00  -1.70e+01  9.77e+00        1    5.01e-06    1.14e-03
   5  8.553131e+02   -7.34e+02    3.61e+02   0.00e+00  -6.32e+00  3.05e-01        1    5.01e-06    1.16e-03
   6  3.306595e+01    8.81e+01    4.10e+02   0.00e+00   1.37e+00  9.16e-01        1    1.29e-05    1.18e-03
   7  6.426770e+00    2.66e+01    1.81e+02   1.29e-01   1.10e+00  2.75e+00        1    1.19e-05    1.20e-03
   8  3.344546e+00    3.08e+00    5.51e+01   3.05e-02   1.03e+00  8.24e+00        1    1.19e-05    1.21e-03
   9  1.987485e+00    1.36e+00    2.33e+01   8.87e-02   9.94e-01  2.47e+01        1    1.19e-05    1.23e-03
  10  1.211585e+00    7.76e-01    8.22e+00   1.05e-01   9.89e-01  7.42e+01        1    1.22e-05    1.25e-03
  11  1.063265e+00    1.48e-01    1.44e+00   6.06e-02   9.97e-01  2.22e+02        1    1.19e-05    1.27e-03
  12  1.056795e+00    6.47e-03    1.18e-01   1.47e-02   1.00e+00  6.67e+02        1    1.22e-05    1.28e-03
  13  1.056751e+00    4.39e-05    3.79e-03   1.28e-03   1.00e+00  2.00e+03        1    1.10e-05    1.30e-03
Ceres Solver Report: Iterations: 14, Initial cost: 1.211734e+02, Final cost: 1.056751e+00, Termination: CONVERGENCE
Initial m: 0 c: 0
Final   m: 0.291861 c: 0.131439
```

从参数值 $m = 0, c=0$开始，初始目标函数值为 $121.173$，Ceres 找到解决方案 $m= 0.291861$， $c = 0.131439$，目标函数值为 $1.05675$。这些值与原始模型 $m=0.3、c= 0.1$ 的参数略有不同，但这是意料之中的。当从噪声数据重建曲线时，我们预计会看到这种偏差。事实上，如果你要评估 $m=0.3、c=0.1$ 的目标函数，拟合效果会更差，目标函数值为 $1.082425$。下图说明了拟合效果。

![least_squares_fit](./pic/least_squares_fit.png)

