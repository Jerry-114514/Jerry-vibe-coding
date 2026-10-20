# AutoDL实验资产恢复指南

此指南对应释放前的内容级备份，不是整个系统盘镜像。原始研究结果保留在三个历史Release，新增缺项和恢复资料保存在[实例释放前备份Release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-instance-retirement-20261020)。最终可否释放，以同目录的验收报告及`release-ready.json`为准。

## 1. 需要保存的入口

- [项目源码与论文](https://github.com/Jerry-114514/Jerry-vibe-coding)：正常克隆仓库即可。
- 原v1：[arp-formal-20261014](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)。
- 7B独立实验：[arp-protocol-v2-20261016](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)。
- 9B及补充对照：[arp-9b-122b-20261017](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-9b-122b-20261017)。
- 新增缺项、环境与完整恢复映射：[arp-instance-retirement-20261020](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-instance-retirement-20261020)。

GitHub自动生成的Source code ZIP只含Git文件，不含Release的原始输出、教师缓存、适配器或恢复状态。四个Release互为补充，不能只下载最新一个然后假定拥有全部实验数据。

## 2. 在新实例恢复

先克隆仓库。然后从释放前Release下载最终审计小包，解压到一个新建的临时目录，找到`restore-map.compact.json`。该映射逐文件记录原路径、大小、SHA256、所需归档及其成员路径。相同内容可能归档在不同路径；恢复工具会映射回原路径，不靠文件名猜测。

在**新实例**上运行下载规划，不需要API密钥：

```bash
python tools/restore_instance_retirement.py plan \
  --map /path/to/restore-map.compact.json \
  --prefix runs/9b-u137-factorial-20261018/ \
  > download-plan.json
```

`download-plan.json`列出实际需要的附件URL、字节数与SHA。将这些附件下载到新实例的同一个目录，例如`/root/autodl-tmp/restore-assets`。不要把完整大包先下载到个人电脑再上传。私有仓库如需授权，使用GitHub正常凭证机制；不要把凭证写进恢复清单或提交到仓库。

恢复选定实验：

```bash
python tools/restore_instance_retirement.py restore \
  --map /path/to/restore-map.compact.json \
  --archives /root/autodl-tmp/restore-assets \
  --destination /root/autodl-tmp/arp \
  --prefix runs/9b-u137-factorial-20261018/
```

可多次传入`--prefix`，分别恢复冻结输入、训练图片和相关脚本；省略`--prefix`才会规划整个盘点范围。恢复工具不联网、不执行归档代码，先验证每段及拼接后归档的SHA，再按成员哈希恢复。它拒绝越界路径和覆盖不同内容的已有文件；已有文件完全相同则跳过。分段归档流式读取，不必再生成一份巨大的拼接包。

若已克隆最新仓库，优先按`runs/`、`data/`等前缀恢复实验资产；实例的源码快照可能与最新README或交接文档不同。需要全量原样恢复时使用另一个空目录，再逐项核对，不要强行覆盖最新代码。

只做评测通常需要适配器、冻结评测清单及图片、精确基础模型、处理器配置和相应环境，不需要加载优化器。继续训练才需要匹配的`resume.pt`、训练清单、教师目标与环境。论文所用固定端点已完成，不应把“恢复能力”误写成继续追加训练预算。

## 3. 重建环境和基础模型

复现资料包保存两个环境的`pip freeze`、Python与系统/驱动信息，以及关键运行库的安装文件完整性核查。两套环境不应合并安装：

| 环境 | Python | PyTorch | Transformers | PEFT |
|---|---|---|---|---|
| 原7B/控制环境 | 3.12.3 | 2.8.0+cu128 | 4.51.3 | 0.15.2 |
| 9B环境 | 3.12.3 | 2.8.0+cu128 | 5.20.1 | 0.21.2 |

在兼容CUDA驱动与Python版本的系统中分别创建虚拟环境，参照对应requirements文件安装，并复核包版本。源码核查只覆盖有wheel RECORD SHA的关键Python/JSON/text文件，不是完整操作系统或二进制镜像证明。

可公开重新下载的预训练基础权重没有重复镜像到此次Release；保存了模型配置、tokenizer、精确身份和已有权重核验信息。Qwen3.5-9B正式revision为`c202236235762e1c871ad0ccb60c8ee5ba337b9a`。恢复时核对冻结环境锁中的权重SHA；不能只根据“9B”或商业模型别名选择替代权重。保留实际运行时的生成配置状态，并使用项目v4.1脚本登记的EOS `[248046,248044]`，不能因新下载文件的默认值不同而悄悄改变测量协议。

原7B原轮与历史v1、9B的精度和处理器约定不同，须按各自冻结配置恢复。运行前先做小样本CPU/模型加载与对应接口检查；论文复核不要求重新跑全部训练。

## 4. 独立于实例的凭证与资源

API密钥、SSH私钥、登录配置、`.bashrc`和实际环境变量值不进入公开归档。将来需要API时重新在新环境配置；现有冻结教师回答可以直接用于复核，无需重付原生成费用。

未备份虚拟环境的可重装二进制目录、公共基础模型大文件、重复导出包及与研究无关的MNIST示例。这不影响已冻结实验产物的内容备份，但意味着无法把本备份当作可直接启动的整机镜像。

现存研究文件逻辑总量约30GB；恢复映射还包含早先移走的恢复状态及原始输出。完整恢复另需这些文件、所选预训练模型、环境和下载归档的空间。实际需求按恢复映射的文件字节数及`download-plan.json`计算；不要仅以压缩包大小估算解压后空间。
