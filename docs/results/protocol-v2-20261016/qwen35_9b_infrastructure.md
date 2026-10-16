# Qwen3.5-9B 并行基础设施准备

按用户要求，在7B最小循环继续运行期间，准备官方 [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 的完整权重和独立环境。该模型官方标识没有额外VL后缀，配置为原生多模态 `Qwen3_5ForConditionalGeneration`。模型结构/加载方式依据 [官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json) 与 [Transformers官方模型文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md)。

- 目标权重：AutoDL `.model_cache/qwen35_9b`。先获取官方仓库固定commit与逐片SHA256；ModelScope作为下载加速源时必须匹配该SHA，不将master标签作为版本保证。可断点分段下载，限制四线程，至少保留12GiB给7B训练存储。
- 独立环境：`/root/autodl-tmp/arp-qwen35-venv`。新环境以只读搜索路径复用既有Torch/CUDA运行时，Transformers/PEFT及新依赖安装在新环境优先路径；约束Torch/torchvision版本，不升级7B活动环境。
- 初始默认pip镜像未提供已锁定的新版Transformers包，另开官方PyPI安装修复任务；保留初始失败记录和修复回执，不把下载中的权重写成完成。
- GPU验收：等待7B最小循环五项通过后，再通过共享GPU锁插入单图BF16有限logits/生成小测试。记录CUDA版本、模型revision、显存峰值和时间。基础设施烟测通过也不等于9B完整LoRA训练显存预检通过。
- 准备状态、锁文件、依赖清单、失败与修复日志及GPU烟测结果直接上传GitHub Release。原始官方权重留在AutoDL，仓库记录固定来源和SHA，不通过本机搬运权重。

当前只授权并行准备9B基础设施；7B独立实验的阶段顺序和固定15次训练预算不变。9B+122B训练需独立冻结模型、输入协议、LoRA匹配范围和预算，不能把7B的token缓存或硬件预检SHA直接当作9B适用证据。
