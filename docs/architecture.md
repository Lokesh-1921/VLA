# Architecture

VisionPilot has two deliberately separate action layers. A VLA produces native continuous robot actions; `ContinuousToDiscreteAdapter` maps those only for the webcam demonstration. BridgeData actions are never relabelled as keyboard commands for core training.

`BaseVLAModel` defines image-plus-instruction prediction. `OpenVLAAdapter` is an optional integration boundary for an official compatible runtime. `NumpySmokePolicy` is a small diagnostic policy that shares the sample and action interfaces, performs actual LoRA-style low-rank head updates, and makes no OpenVLA equivalence claim.

The model-to-actuator boundary is `ActionExecutor`. The default executor is simulated and rate limited. The module contains no serial, GPIO, or robot transport code.
