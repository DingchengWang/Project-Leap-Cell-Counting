# Project Leap Cell Counting

[中文](README_中文.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/)

Cell Counting is a **Project Leap** module for reviewing and counting candidate cells in multichannel fluorescence TIFF images. It uses DAPI and a separate support-marker channel to detect candidate nuclei and bodies, associate them where the image evidence supports a unique match, and present the resulting cell records for inspection on the original Z planes.

Researchers can review the proposed records, add missed cells, merge duplicate records, and exclude incorrect detections before saving the count. The result includes a numbered image, a per-cell CSV table, and a session that preserves the observations and edits for later review. Accepted records can contain a nucleus, an independently detected body, or a uniquely associated pair.

This release provides module **1.0.0** for **Project Leap v1.0.0**. Download the main application from the [Project Leap download page](https://dingchengwang.github.io/Project-Leap/en/downloads/).

## Use the module

Install Cell Counting from the module page inside Project Leap. The application manages the module, shared scientific runtime, and fixed Cellpose model.

To start, import a TIFF, choose the image series and time point when needed, and confirm the DAPI channel, a different support channel, and its marker. The user guide takes you from these choices through review and saving.

- [User guide](docs/guide.en.md): input selection, review controls, saving, reopening, and output columns.
- [Methods](docs/methods.en.md): local Z detection, original-grid recovery, support-body detection, and cell identity.
- [Cell identity and persistence contract](docs/prototype-d/CELL_IDENTITY.md): exact record and label-file conventions.

## Source and licensing

The counting engine is in [`engine/leap_cell_counting`](engine/leap_cell_counting); [`native/counting_stats.c`](native/counting_stats.c) computes source-image statistics. The desktop interface, module installation, and result editor belong to the [Project Leap application](https://github.com/dingchengwang/Project-Leap).

Original Project Leap code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE). Third-party software and model weights retain their own licenses. The fixed model comes from [MouseLand's Cellpose project](https://github.com/MouseLand/cellpose) and its [official `cpsam_v2` model source](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2). See the application's [third-party notices](https://github.com/DingchengWang/Project-Leap/blob/main/THIRD_PARTY_NOTICES.md) for sources and attribution.
