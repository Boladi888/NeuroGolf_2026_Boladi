Live bar: 222, SHA-256 5fed3434f485dc5f2c65843c938d93b364e044b5983203ef7f8a8a5ef39ba5ef.
Attempted route: remove shared `q_zero` initializer by omitting QLinearMatMul/QLinearConv zero-point inputs.
Failed gate: schema check before build; ONNX opset-14 schemas mark all QLinear zero-point inputs as FormalParameterOption.Single.
Why dead: zero-points are required inputs for both QLinearMatMul and QLinearConv in this environment, so omission is invalid rather than a score shave.
Remaining hypothesis: only a different non-QLinear output formulation could remove q_zero, but current int8 QLinear path is likely cheaper than ConvInteger/MatMulInteger alternatives.
