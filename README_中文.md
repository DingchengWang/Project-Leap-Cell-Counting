# Project Leap Cell Counting

[English](README.md) · [Project Leap 主站](https://dingchengwang.github.io/Project-Leap/zh/)

Cell Counting 是 **Project Leap** 中的细胞计数模块，供研究者在多通道荧光 TIFF 图像中识别候选细胞、逐层审核并确认计数。模块分别从 DAPI 通道识别核、从另一辅助标记物通道识别胞体，并依据图像证据匹配唯一对应的核与胞体，供用户在原始 Z 切片上核对。

审核时可以补加遗漏细胞、合并重复记录、排除错误检测，再保存计数。结果包括编号图、逐细胞 CSV 表格和可重新打开的编辑会话，保留检测依据与编辑记录。独立核、独立胞体或唯一配对的核与胞体均可组成细胞记录，最终按接受状态计入总数。

本次提供模块 **1.0.0**，适用于 **Project Leap v1.0.0**。主程序下载见 [Project Leap 下载页面](https://dingchengwang.github.io/Project-Leap/zh/downloads/)。

## 使用模块

在 Project Leap 内打开模块页面并安装 Cell Counting。软件统一管理模块、共享科学运行环境和固定 Cellpose 模型。

导入 TIFF 后，按需要选择图像组和时间点，再确认 DAPI 通道、与其不同的辅助通道及标记物。后续分析、审核与保存操作见使用指南。

- [使用指南](docs/guide.zh.md)：输入选择、审核操作、保存、重开及输出列说明。
- [方法说明](docs/methods.zh.md)：局部 Z 检测、原始网格恢复、辅助通道胞体识别和细胞身份。
- [细胞身份与持久化契约](docs/prototype-d/CELL_IDENTITY.md)：记录字段和标签文件的精确定义。

## 源码与许可

计数引擎位于 [`engine/leap_cell_counting`](engine/leap_cell_counting)，[`native/counting_stats.c`](native/counting_stats.c) 负责原图统计。桌面界面、模块安装及结果编辑器属于 [Project Leap 主软件](https://github.com/dingchengwang/Project-Leap)。

Project Leap 原创代码采用 [Apache-2.0](LICENSE) 许可，署名见 [NOTICE](NOTICE)。第三方软件和模型权重保留各自许可。固定模型来源为 [MouseLand 的 Cellpose 项目](https://github.com/MouseLand/cellpose)及其[官方 `cpsam_v2` 模型](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)；来源与署名见主软件的[第三方声明](https://github.com/DingchengWang/Project-Leap/blob/main/THIRD_PARTY_NOTICES.md)。
