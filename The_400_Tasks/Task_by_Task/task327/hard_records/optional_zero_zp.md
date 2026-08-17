live bar: 268, sha 01ffc402949dce588c2e48584704325124f0355b57a7ffac798664e394b279eb
route: omit QLinearConv zero-point inputs that are mathematically default zero and delete the scalar zero initializer.
failed build/check: `python The_400_Tasks/Task_by_Task/task327/prototypes/build_task327_omit_zero_zp.py` -> `ValidationError: Node (v_n2)'s input 2 is marked single but has an empty string in the graph`
why dead: this ONNX/ORT schema requires the QLinearConv input zero-point tensor; the scalar zero initializer remains load-bearing.
remaining hypothesis: only w_zero/y_zero may be optional, but x_zero still references the scalar, so that variant cannot beat the 268 live bar.
