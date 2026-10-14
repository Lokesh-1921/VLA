# Training protocol

1. Run `check_system.py` and `benchmark_model.py --backend smoke`.
2. Run `train.py --config configs/mac_mlx.yaml --smoke`; confirm a checkpoint reloads.
3. Inspect a small real dataset slice and run a model-specific data/load/inference test.
4. Train only after `allow_large_run: true` is consciously recorded in a copied run configuration.

The run manifest records configuration, device information, package availability, timings, dataset count, seed, and checkpoint paths. Runs are uniquely named and resume requires an explicit checkpoint path. The NumPy trainer is a pipeline test, not the intended OpenVLA trainer. Production OpenVLA LoRA needs an official compatible backend and should write the same manifest fields.
