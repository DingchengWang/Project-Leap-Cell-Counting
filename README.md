# Project Leap Cell Counting

[中文](README_中文.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/)

Cell Counting is a **Project Leap** module for spatial localization and counting of astrocytes in multichannel fluorescence TIFF images. It uses DAPI to locate nuclei and a separate, user-defined auxiliary fluorescence channel to identify bodies and assess perinuclear signal. Locating cells throughout a three-dimensional image stack helps distinguish cells that overlap in a two-dimensional projection.

Researchers can inspect cells across the original Z planes, add missed cells, merge duplicates, and delete incorrect detections before confirming the count. Outputs include a numbered fluorescence overlay, a per-cell CSV table, and an editable session for later review.

Cell Counting **1.0.0** runs within **Project Leap v1.0.0**. Download the main application from the [Project Leap download page](https://dingchengwang.github.io/Project-Leap/en/downloads/).

## Use the module

In Project Leap, open **Modules & Maintenance**, find **Cell Counting**, and click **Install analysis module**. Once installation finishes, open **Cell Counting** from the sidebar, then follow the user guide to select an image, run the analysis, and review the count.

- [User guide](docs/guide.en.md): input selection, review controls, saving, reopening, and output columns.
- [Methods](docs/methods.en.md): local Z detection, original-grid recovery, support-body detection, and cell identity.
- [Cell records and label files](docs/prototype-d/CELL_IDENTITY.md): exact field definitions and file conventions.

## Source and licensing

The counting engine is in [`engine/leap_cell_counting`](engine/leap_cell_counting); [`native/counting_stats.c`](native/counting_stats.c) computes source-image statistics. The desktop interface, module installation, and result editor belong to the [Project Leap application](https://github.com/dingchengwang/Project-Leap).

Original Project Leap code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE). Third-party software and model weights retain their own licenses. The fixed model comes from [MouseLand's Cellpose project](https://github.com/MouseLand/cellpose) and its [official `cpsam_v2` model source](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2). See the application's [third-party notices](https://github.com/DingchengWang/Project-Leap/blob/main/THIRD_PARTY_NOTICES.md) for sources and attribution.
