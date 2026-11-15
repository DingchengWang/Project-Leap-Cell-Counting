# Project Leap Cell Counting

[English](README.md) · [Project Leap 主站](https://dingchengwang.github.io/Project-Leap/zh/)

Cell Counting 是 **Project Leap** 内的细胞计数模块，用于多通道荧光 TIFF 图像。模块分别识别 DAPI 核和所选辅助通道中的胞体，再匹配为细胞记录，供用户逐层审核和调整计数。

本次提供模块 **1.0.0**，适用于 **Project Leap 1.0.0，build 1**。主程序 DMG 的公开下载仍在准备中。

## 使用模块

在 Project Leap 内打开模块页面并安装 Cell Counting。软件负责安装共享科学运行环境和固定 Cellpose 模型；本仓库不提供独立手动安装器。

导入 TIFF，按需要选择图像组和时间点，确认 DAPI 及与其不同的辅助通道，然后开始分析。分析完成后，在原始 Z 切片上审核对象，调整计数并确认保存，即可得到编号图、CSV 表格和可继续编辑的会话。

- [使用指南](docs/guide.zh.md)：输入选择、审核操作、保存、重开及输出列说明。
- [方法说明](docs/methods.zh.md)：局部 Z 检测、原始网格恢复、辅助通道胞体识别和细胞身份。
- [细胞身份与持久化契约](docs/prototype-d/CELL_IDENTITY.md)：记录字段和标签文件的精确定义。

## 源码与许可

计数引擎位于 [`engine/leap_cell_counting`](engine/leap_cell_counting)，[`native/counting_stats.c`](native/counting_stats.c) 负责原图统计。桌面界面、模块安装及结果编辑器属于 [Project Leap 主软件](https://github.com/dingchengwang/Project-Leap)。

Project Leap 原创代码采用 [Apache-2.0](LICENSE) 许可，署名见 [NOTICE](NOTICE)。第三方软件和模型权重保留各自许可。固定模型来源为 [MouseLand 的 Cellpose 项目](https://github.com/MouseLand/cellpose)及其[官方 `cpsam_v2` 模型](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)；来源与署名见主软件的[第三方声明](https://github.com/DingchengWang/Project-Leap/blob/main/THIRD_PARTY_NOTICES.md)。
