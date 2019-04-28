# NaiveCompGraph
Naive Computation Graph.

## Stage 1
```
cd examples/stage1
./run.sh <example_id>
```

Alternative using Makefile:
```
cd examples/stage1
make
./main < data/<example_id>.txt
```

## Examples
建议阅读顺序（除了stage1的代码需要指定example id，其他所有示例直接运行`./run.sh`即可观察结果）：

1. `examples/test_dtype/` 理解数据类型（data type系统）。
2. `examples/test_tensor` 理解Tensor类型，包括定义，shape，取值。
3. `examples/test_arith` 理解Op系统，学会创建一个Op（OpAdd）和一个OpContext（执行上下文），进行运算。
4. `examples/test_graph` 理解Graph系统，学会用`graph.op<OpType>(name, desc, inputs...)`创建Op，用`GraphForwardContext`进行Eval。
5. `examples/stage1` 实际代码。
6. `examples/stage1/print_op.h`, `examples/stage1/cond_op.h`，定义自己的Op。

## Manual

结构：
```
core/ 包含基础定义文件(core.h datatype.h), tensor的实现(tensor.h, tensor.cc), op基础(op.h, op.cc).
ops/ 包含基本op，目前只有基础算数运算(e.g., +, -, *, /)
graph/ 包含图定义相关。包括图op基础(graph_op.h, graph_op.cc), 拓扑排序相关(graph_forward_impl.h), 和图op(graph/ops/), 目前只有占位符、常量、变量和基础算术运算。
```

Tensor：

1. `Tensor`是存放数据的基本单位。
2. `TensorDesc`定义了Tensor的shape（标量是0维，shape={}, 向量是1维，shape={n}，矩阵是2维，shape={n, m}）。
3. `TensorStorage`存放数据。
4. 使用时直接使用`TensorPtr=std::shared_ptr<Tensor>`。
5. 获取数据使用`tensor_ptr->as<DTypeName::Float32>()->data_ptr()` (返回`const float *`)或者`tensor_ptr->as<DTypeName::Float32>()->mutable_data_ptr()` (返回`float *`)。
6. 可以使用`tensor(DTypeName::Float32, {})`创建空tensor，第二个参数是shape。可以使用`scalar(DTypeName::Float32, 1.0)`创建数值内容为1.0，数据类型为float32的标量tensor。

Op:

1. `Op`是关于Tensor的操作，使用时需要重载两个函数`check_inputs`和`compute`。
2. `check_inputs(OpContext &ctx, const TensorVec &inputs)`检查用户输入的tensors，`TensorVec=std::vector<TensorPtr>`。
3. `TensorVec compute(OpContext &ctx, const TensorVec &inputs)`返回计算结果。
4. `Op`所有的运算都是真实的基于数据的运算，和计算图无关。

Graph:
1. `GraphTensor`是定义计算图使用的Tensor类型(类比Tensor)。只有Shape和数据类型信息，没有实际数据。
2. `GraphOp`是定义计算图使用的Op类型，需要调用实际的`Op`才能进行运算。
