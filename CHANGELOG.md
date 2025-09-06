## Changelog

| Version   | Changelog                                                                                                                                                                                                                                                    |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **0.0.1** | Init version. Model is able to learn. Installation in system/user-local packages. Training application is available as relgat-train CLI. With model state additional files are also saved: relation to index mapping and learning-run config.                |
| 0.1.0     | Fixed Inf/Nan problems during training, refactoring whole training code. Added base models, architectures definitions, base model trainer and training app. Apps are available under `relgat_apps` dir. Integration with `rdl_mlm_utils`.                    |  
| 0.2.0     | Added projection layer. MultiObjectiveLoss implementation. Final embedding are produces in the same space as base embedding before transformation. CLI RelGAT training script renamed to `relgat-projector-train`. Renamed main module as `relgat_projector` | 
