# numerical-optimization
数值优化算法



### 目录

#### 解析算法

* 线性搜索（Linear Search）<a href="#线性搜索（Linear Search）">click</a>
* 置信域（Trust Region, TR）<a href="#置信域搜索（Trust Region, TR）">click</a>

#### 数值算法

* 遗传算法（Genetic Algorithm, GA）<a href="#遗传算法（Genetic Algorithm, GA）">click</a>
* 粒子群算法（Particle Swarm Algorithm, PSO）<a href="#粒子群算法（Particle Swarm Optimization, PSO）">click</a>

***

### 算法背景

求解（可能在一定条件下）使得$f(x)$取得最优值（一般是最小值）的$x$，即$x_{\rm opt}$，记为：


$$
\begin{aligned}
\min & \quad f(x) \\
s.t. & \quad g_0(x) < 0 \\
& \quad g_1(x) \leq 0 \\
& \quad ... \\
\end{aligned}
$$


其中自变量$x$为一维标量或多维向量，记维数为$N$，$f(x)$输出一般是一维标量。

在进行优化的过程中常常会涉及到对函数$f$在$x$处的Jacobian矩阵和Hessian矩阵的计算：

Jacobian矩阵：


$$
J = \begin{aligned}
	\left[
		\begin{array}{c}
		&\frac{\partial f_0}{\partial x_0}, &\frac{\partial f_0}{\partial x_1}, &..., &\frac{\partial f_0}{\partial x_{N - 1}} \\
		&\frac{\partial f_1}{\partial x_0}, &\frac{\partial f_1}{\partial x_1}, &..., &\frac{\partial f_1}{\partial x_{N - 1}} \\
		&...,&...,&...,&... \\
		&\frac{\partial f_{M-1}}{\partial x_0}, &\frac{\partial f_{M-1}}{\partial x_1}, &..., &\frac{\partial f_{M-1}}{\partial x_{N - 1}} \\
		\end{array}
	\right]
\end{aligned}
$$


Hessian矩阵：


$$
H = \begin{aligned}
	\left[
		\begin{array}{c}
		&\frac{\partial^2 f}{\partial x_0x_0}, &\frac{\partial^2 f}{\partial x_0x_1}, &..., &\frac{\partial^2 f}{\partial x_0x_{N - 1}} \\
		&\frac{\partial^2 f}{\partial x_1x_0}, &\frac{\partial^2 f}{\partial x_1x_1}, &..., &\frac{\partial^2 f}{\partial x_1x_{N - 1}} \\
		&...,&...,&...,&... \\
		&\frac{\partial^2 f}{\partial x_{N-1}x_0}, &\frac{\partial^2 f}{\partial x_{N-1}x_1}, &..., &\frac{\partial^2 f}{\partial x_{N-1}x_{N - 1}} \\
		\end{array}
	\right]
\end{aligned}
$$


***Note***：

* 本算法中已经实现了使用数值方法计算Jacobian矩阵和Hessian矩阵的功能，参见*mod.mathematics.partial_derives*中的*NumPartialDerives*

***

### 解析算法

##### 线性搜索（Linear Search）

线性搜索方法是优化算法中最为简单的一种方法。其主要计算步骤如下：

* 设定初始解$x_{\rm init}$、最大迭代次数$iter_{\rm max}$、迭代精度$tol_x$

* 计算函数$f$在当前解$x$处的梯度$grad_x$，即Jacobian矩阵$J$的第一行向量，取负梯度方向为$x$的下一步迭代方向:


  $$
  direc_x = -grad(x)
  $$


* 获得了$x$的迭代方向$direc_x$后，设定在此方向前进长度$\alpha$，那么$x$的下一步迭代值为：


  $$
  x_{\rm new} = x + \alpha \cdot direc(x)
  $$
  同时获得下一步迭代函数值为：


  $$
  y_{\rm new} = f(x + \alpha \cdot direc(x))
  $$
  可以看出，$y_{\rm new}$为$\alpha$的函数，而最优的$\alpha$应满足使得对应的函数值最低，即：


  $$
  \alpha^* = {\mathop{\arg\min}_\alpha} f(x + \alpha \cdot direc(x))
  $$


* 获得了迭代步长$\alpha^*$后，更新$x$：


  $$
  x_{\rm new} = x + \alpha^* \cdot direc(x)
  $$


* 判定是否满足迭代终止条件，若满足则终止迭代并输出结果，否则继续迭代：


  $$
  \begin{aligned}
  &\text{if } \|x_{\rm new} - x\| \leq tol_x: \\
  &\quad\quad \text{break and output } x \\
  &\text{else }: \\
  &\quad\quad x = x_{\rm new}
  \end{aligned}
  $$


***Note***:

* 相关算法细节参见*lib.linear_search*


##### 置信域搜索（Trust Region, TR）

信赖域算法TR可以用来求解非线性规划问题（NLP, NonLinear Programing），比如含二次项问题的优化求解。

在第$k$步迭代时，将函数$f$在$x_k$处展开有：


$$
f(x_k+p) = f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2f(x_k)p + o(p^2)
$$


其中，向量$p_{k, N \times1}$为该步的局部摄动向量。将上述表达式使用二次型$m$和$p$进行局部近似有：


$$
m(p_k) = f(x_k) + J(x_k)p_k + \frac{1}{2}p_k^TH(x_k)p_k
$$


其中$J(x_k)$、$H(x_k)$分别为函数$f$在$x_k$处的Jacobian矩阵和Hessian矩阵。可以使用有限差分或拟牛顿法来对Hessian矩阵进行近似，请参见*mod.mathematics.partial_derives*中的*NumPartialDerives*中对Hessian矩阵计算的内容。这样，我们的优化目标为在$x_k$附近的置信域$R_k$中寻找迭代向量$p_k$使得$m(p_k)$取得极小值：


$$
\begin{aligned}
&\min \quad m(p_k) = f(x_k) + J(x_k)p_k + \frac{1}{2}p_k^TH(x_k)p_k \\
&s.t. \quad \|p_k\| \leq h_k
\end{aligned}
$$


这里$h_k$为$x_k$附近信赖域上界，或称为信赖域半径。接下来的计算中我们需要确定置信域边界，可以分别计算出当前迭代下的实际优化量和使用近似计算出的优化量：


$$
\begin{aligned}
\Delta f(x_k) &= f(x_k) - f(x_k + p_k) \\
\Delta m(p_k) &= f(x_k) - m(p_k)
\end{aligned}
$$


定义实际优化量和预测优化量的比值$r_k$用于衡量二次模型与目标函数的近似程度：


$$
r_k = \frac{\Delta f(x_k)}{\Delta m(p_k)}
$$


显然$r_k$越接近1越好，在实际迭代过程中：

* 当$r_k \leq 0.25$, 说明步子迈得太大了，应缩小信赖域半径：$h_{k+1} = \frac{\|p_k\|}{4}$；
* 当$r_k \geq 0.75$且$\|p_k\| = h_k$，说明这一步已经迈到了信赖域半径的边缘，并且步子有点小，可以尝试扩大信赖域半径：$h_{k+1} = 2h_k$；
* 当$0.25 \leq r_k \leq 0.75$，说明这一步迈出去之后，处于“可信赖”和“不可信赖”之间，可以维持当前的信赖域半径：$h_{k+1} = h_k$；
* 当$r_k < 0$，说明函数值是向着上升而非下降的趋势变化了（与最优化的目标相反），这说明这一步迈得错得“离谱”了，这时不应该走到下一点，而应“原地踏步”，即$x_{k+1} = x_{k}$，并且和上述$0.25 \leq r_k \leq 0.75$一样，缩小信赖域；反之，在$r_k>0$的情况下，都可以走到下一点，即$x_{k+1} = x_k + p_k$



***Note***:

* 相关算法细节参见*lib.trust_region*

***

### 数值算法

##### 遗传算法（Genetic Algorithm, GA）





##### 粒子群算法（Particle Swarm Optimization, PSO）



***

### TODOs
