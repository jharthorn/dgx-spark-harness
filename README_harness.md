# DGX Spark Harness Skeleton

This directory follows Section 7 of Test_Plan_v3.0.md and houses the stack configs, inputs, source runners, workloads, and telemetry tools referenced throughout the plan.

- `configs/`: Stack A vs Stack B YAML configs (Section 4)
- `inputs/`: Prompt and LoRA assets used across workloads (Section 3)
- `src/`: Load generator, workloads, telemetry collectors, and runners for H0-H8B (Sections 5-9)
- `analysis/`: Post-processing utilities per Appendix C

Refer to Test_Plan_v3.0.md for full instructions on how these components integrate.
