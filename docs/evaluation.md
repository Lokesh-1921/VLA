# Evaluation

Fix the evaluation samples and random seed before comparing baseline, Bridge fine-tuning, and custom adaptation. Use the same normalizer, action dimension, split, and task protocol for every row.

For continuous actions, VisionPilot reports MAE and RMSE. For discrete demo adapters it reports accuracy, macro precision, macro recall, macro F1, and a confusion matrix. Inference benchmark output includes loading time, image preprocessing time, action latency, throughput, and process-memory delta. Task success is included only when supplied by a genuine environment or human-labelled protocol.
