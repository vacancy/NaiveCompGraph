# NaiveCompGraph
Naive Computation Graph.

## Implemented Features
- (Bonus 3) Tensor (Max Dimension = 15).
- Basic tensor operations: unary arithmetic, binary arithmatic, matrix multiplication, reshaping and broadcasting, slicing, indexing, reducing (min, max, sum, mean).
- Reshaping, broadcasting and slicing are implemtented using the stride trick. No actual data copy needed. Tensor are made contiguous only when necessary.
- Most operations supports non-contiguous input (e.g., except matmul, for performance perpose). Many operations (e.g., arithmatic operations) are optimized when the input view is contiguous.
- Complete (and dynamic) data type support.
- Computation graph.
- (Bonus 4) Session for storing shared tensors (i.e., variables in Tensorflow or Parameters/Buffers in PyTorch).
- (Bonus 1) Assign Op for updating variables.
- (Bonus 2) Gradient for all operations are implemented.
- Second-order gradient is supported.
- Graph operations allow dynamic shapes. E.g., `G::reshape(x, G::shape_cat({x.shape(0), -1}))`. Note that `x.shape(0)` returns a graph tensor (an int64-typed scalar).
- (Bonus 6) Complete MNIST example.

## MNIST Example
构建了一个两层的神经网络：
Flatten input: `(batch_size, 784)` -> Hidden layer `(batch_size, 512)` -> Logits `(batch_size, 10)`.
网络使用Tanh激活。SGD learning rate = 0.01。网络初始化用Normal(0, 0.01)初始化Weights，全0初始化Bias。

网络结构部分代码：
```
    MLPModel(std::mt19937 &rng) : rng(rng) {
        image = G::placeholder("image", {100, 784}, DTypeName::Float32);
        label = G::placeholder("label", {100}, DTypeName::Int64);
        linear1 = G::linear("linear1", image, 512, rng);
        activation1 = G::tanh(linear1);
        logits = G::linear("linear2", activation1, 10, rng);
        pred = logits.max(-1)[1];

        prob = G::softmax(logits, -1);
        loss = G::xent_sparse(prob, label, -1).mean(0);
        accuracy = (pred.eq(label)).float32().mean(0);
    }
```

Linear Layer初始化
```
GTensorPtr linear(std::string name, GTensorPtr x, ssize_t output_dim, std::mt19937 &rng, double stddev) {
    auto W = variable(name + ":W", ::ncg::rand_normal(rng, x->desc().dtype(), {x->desc().shape(1), output_dim}, 0, stddev));
    auto b = variable(name + ":b", ::ncg::zeros(x->desc().dtype(), {output_dim}));
    return matmul(x, W) + b.unsqueeze(0);
}
```

SGD部分代码：
```
    GTensorVec train_ops(float lr=0.01) {
        GTensorVec ops;
        auto &graph = get_default_graph();

        graph.backward(loss);
        for (const auto &name : {"linear1:W", "linear2:W", "linear1:b", "linear2:b"}) {
            auto W = graph.find_op(name)->outputs()[0];
            auto G = W->grad(loss);
            auto new_W = W - G * lr;
            ops.push_back(G::assign(W, new_W));
        }

        return ops;
    }
```

训练50Epoch后可以达到准确度97.5%。

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

## Stage 2
```
cd examples/stage2
make
./main < data/<example_id>.txt
```

## Newton Method
```
cd examples/newton_method
make
./main < in.txt
```

# MNIST MLP
```
cd examples/mnist
./download-data.sh
make
./main
```

## Tutorial Examples
建议阅读顺序（除了stage1的代码需要指定example id，其他所有示例直接运行`./run.sh`即可观察结果）：

1. `examples/1_test_dtype` 理解数据类型（data type系统）。
1. `examples/2_test_tensor` 理解Tensor类型，包括定义，shape，取值。
1. `examples/2_test_tensor_pickle` 理解数据持久化（Pickle,Unpickle）。
1. `examples/3_test_op_arith` 理解Op系统，学会创建一个Op（OpAdd），进行运算。
1. `examples/3_test_op_shape` 深入理解Shape, Axes，学习Reshape，Permute, Expand操作。
1. `examples/3_test_op_slice` 理解Slice操作，包括Narrow（即Python Slice），IndexSelect和Gather。
1. `examples/3_test_op_reduce` 理解各种reduce操作，比如`reduce_sum`。
1. `examples/4_test_graph_arith` 理解Graph系统，学会用`G::op_name`创建Op，用`GraphForwardContext`进行Eval。
1. `examples/4_test_graph_matrix` 理解Graph系统，进行矩阵运算。
1. `examples/stage1/print_op.h`, `examples/stage1/cond_op.h`，定义自己的Op。

## Manual

结构：
```
core/ 包含基础定义文件(core.h datatype.h), tensor的实现(tensor.h, tensor.cc), op基础(op.h, op.cc).
ops/ 包含基本op，目前只有基础算数运算(e.g., +, -, *, /)
graph/ 包含图定义相关。包括图op基础(op.h/cc, tensor.h/cc), 拓扑排序相关(graph.h, graph.cc), 和图op(graph/ops/)。
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

## Officially Supported Ops
```
// Defined in graph/tensor.h

// elemwise::misc
GTensorPtr cast(TensorPtr a, DTypeName dtype);
GTensorPtr cond(TensorPtr a, TensorPtr b, TensorPtr c);

// elemwise::unary
GTensorPtr neg(GTensorPtr a);
GTensorPtr sin(GTensorPtr a);
GTensorPtr cos(GTensorPtr a);
GTensorPtr tan(GTensorPtr a);
GTensorPtr log(GTensorPtr a);
GTensorPtr exp(GTensorPtr a);
GTensorPtr tanh(GTensorPtr a);
GTensorPtr sigmoid(GTensorPtr a);
GTensorPtr reciprocal(GTensorPtr a);

GTensorPtr add(GTensorPtr a, GTensorPtr b);
GTensorPtr sub(GTensorPtr a, GTensorPtr b);
GTensorPtr mul(GTensorPtr a, GTensorPtr b);
GTensorPtr div(GTensorPtr a, GTensorPtr b);
GTensorPtr ge(GTensorPtr a, GTensorPtr b);
GTensorPtr le(GTensorPtr a, GTensorPtr b);
GTensorPtr geq(GTensorPtr a, GTensorPtr b);
GTensorPtr leq(GTensorPtr a, GTensorPtr b);
GTensorPtr eq(GTensorPtr a, GTensorPtr b);
GTensorPtr neq(GTensorPtr a, GTensorPtr b);
GTensorPtr pow(GTensorPtr a, GTensorPtr b);
GTensorPtr min(GTensorPtr a, GTensorPtr b);
GTensorPtr max(GTensorPtr a, GTensorPtr b);

// netsrc
GTensorPtr placeholder(std::string name, const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr constant(TensorPtr value);
GTensorPtr variable(std::string name, TensorPtr init_value);
GTensorPtr zeros(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr ones(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);

// linalg
GTensorPtr matmul(GTensorPtr a, GTensorPtr b, bool transpose_a=false, bool transpose_b=false);

// update
GTensorPtr assign(GTensorPtr a, GTensorPtr b);

// reduce
GTensorVec reduce_min(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorVec reduce_max(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorPtr reduce_sum(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorPtr reduce_mean(GTensorPtr a, ssize_t axis, bool keepdims=false);

// shape
GTensorPtr reshape(GTensorPtr a, const ShapeVec &shape);
GTensorPtr permute(GTensorPtr a, const ShapeVec &axes);
GTensorPtr expand(GTensorPtr a, const ShapeVec &shape);
GTensorPtr squeeze(GTensorPtr a, ssize_t axis);
GTensorPtr unsqueeze(GTensorPtr a, ssize_t axis);

// shape
GTensorPtr shape_of(GTensorPtr a);
GTensorPtr shape_of(GTensorPtr a, ssize_t axis);
GTensorPtr shape_cat(const GTensorVec &a);

// slice
GTensorPtr concat(const GTensorVec &a, ssize_t axis);
GTensorVec split(GTensorPtr a, ssize_t axis, const ShapeVec &splits);
GTensorPtr narrow(GTensorPtr a, ssize_t axis, ssize_t start, ssize_t length);
GTensorPtr index_select(GTensorPtr a, ssize_t axis, GTensorPtr b);
GTensorPtr gather(GTensorPtr a, ssize_t axis, GTensorPtr b);
```
