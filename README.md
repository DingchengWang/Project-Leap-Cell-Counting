# Project Leap Cell Counting

[中文](README_中文.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/)

Cell Counting is a module for reviewing and counting cells in multichannel fluorescence TIFF images within **Project Leap**. It combines DAPI nuclear observations with independently detected bodies in a selected support channel, then presents the resulting cell records for slice-by-slice review and editing.

This release provides module **1.0.0** for **Project Leap 1.0.0 (build 1)**. Download the main application from the [Project Leap download page](https://dingchengwang.github.io/Project-Leap/en/downloads/).

## Use the module

Install Cell Counting from the module page inside Project Leap. The application manages the shared scientific runtime and fixed Cellpose model. This repository does not provide a separate manual installer.

Import a TIFF, choose the image series and time point when needed, confirm DAPI and a different support channel, and start analysis. Review the proposed cells on the original Z slices, adjust the count, and confirm to save a numbered image, a CSV table, and a recoverable editing session.

- [User guide](docs/guide.en.md): input selection, review controls, saving, reopening, and output columns.
- [Methods](docs/methods.en.md): local Z detection, original-grid recovery, support-body detection, and cell identity.
- [Cell identity and persistence contract](docs/prototype-d/CELL_IDENTITY.md): exact record and label-file conventions.

## Source and licensing

The counting engine is in [`engine/leap_cell_counting`](engine/leap_cell_counting); [`native/counting_stats.c`](native/counting_stats.c) computes source-image statistics. The desktop interface, module installation, and result editor belong to the [Project Leap application](https://github.com/dingchengwang/Project-Leap).

Original Project Leap code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE). Third-party software and model weights retain their own licenses. The fixed model comes from [MouseLand's Cellpose project](https://github.com/MouseLand/cellpose) and its [official `cpsam_v2` model source](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2). See the application's [third-party notices](https://github.com/DingchengWang/Project-Leap/blob/main/THIRD_PARTY_NOTICES.md) for sources and attribution.
