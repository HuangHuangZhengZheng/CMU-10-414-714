从数学的角度推导 Summation 的梯度，我们可以使用链式法则。

$$
设 f(x) = \sum_{i=1}^{n} x_i，其中 x = (x_1, x_2, ..., x_n)。

对于任意 j \in \{1, 2, ..., n\}，我们有：

\begin{align*}
\frac{\partial f}{\partial x_j} &= \frac{\partial}{\partial x_j} \sum_{i=1}^{n} x_i \\
&= \frac{\partial}{\partial x_j} (x_1 + x_2 + ... + x_n) \\
&= \frac{\partial x_1}{\partial x_j} + \frac{\partial x_2}{\partial x_j} + ... + \frac{\partial x_n}{\partial x_j}
\end{align*}

根据链式法则，对于任意 i \neq j，有 \frac{\partial x_i}{\partial x_j} = 0。而对于 i = j，有 \frac{\partial x_j}{\partial x_j} = 1。

因此，我们得到：

\begin{align*}
\frac{\partial f}{\partial x_j} &= \frac{\partial x_1}{\partial x_j} + \frac{\partial x_2}{\partial x_j} + ... + \frac{\partial x_n}{\partial x_j} \\
&= 0 + 0 + ... + 1 + ... + 0 \\
&= 1
\end{align*}

也就是说，对于任意 j，\frac{\partial f}{\partial x_j} = 1。
$$

这意味着，对于任意输入张量 x，Summation 的梯度都是一个与 x 形状相同的张量，其中所有元素都是 1。


在代码中，我们可以通过广播和重塑 out_grad 来实现这个梯度计算。具体步骤如下：

1. 获取输入张量的形状 shape。
2. 初始化输出形状 shape_out 为全 1 的列表。
3. 如果指定了 axes，使用指定的 axes；否则，默认对所有维度求和。
4. 遍历 shape，如果维度不在 axes 中，则将 shape_out 中对应的元素设为 out_grad 的相应维度的大小。
5. 将 out_grad 重塑为 shape_out 的形状。
6. 使用 broadcast_to 将重塑后的 out_grad 广播到与输入张量相同的形状。

这样就得到了 Summation 的梯度。