# ARP on AutoDL: B–E execution protocol

The source reference is `docs/idea.std.zh.pdf` (M0–M4, equations 1–5).
The supplied checklist is preserved in `docs/experiment_checklist.source.md`.
Phase-A inputs stay frozen; their original byte hashes are authoritative.

## Remote environment

- Instance: `root@connect.bjb1.seetacloud.com`, SSH port `34903`.
- Project: `/root/autodl-tmp/arp`.
- Venv: `/root/autodl-tmp/arp-venv` (Python 3.12.3).
- Image's PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128 are reused.
- Remaining exact versions: `requirements-experiment.txt`.
- InternVL's Qwen language attention uses SDPA with fp16 weights. The upstream eager implementation produced non-finite attention on the real 2B preflight input; the same checkpoint passed after the kernel change. A logits processor stops on non-finite generation scores for every model. Old preflight ledgers are archived when the numerical backend changes; no such ledger can enter formal analysis.
- Public model code/tokenizers come from frozen HF revisions in `configs/models.lock.json`.
- Large weight downloads may use ModelScope; **every weight must match the SHA256 of the pinned HF file**. A different model version cannot be substituted.
- InternVL 2B/8B and LLaVA 0.5B weights live on the system disk in `/root/arp-model-cache`; others live on the data disk under `.model_cache`. Each download preserves 5 GiB on its target disk. Models load serially on the one GPU.

## What is fixed before formal inference

`configs/experiment.json` records every implementation convention absent from the short standard card:

- 120 nominal settings: absolute/relative drop × uncorrected/corrected floor × none/raw-family/corrected-family headroom × rho = 0, .25, .5, .75, 1 × parse failure wrong/exclude.
- Adjacent-edge gaps are sorted descending. For two edges OWA weights are `[rho, 1-rho]`; for one edge OWA is necessarily `[1]`.
- Headroom is `(x_L-x_s)/(u-x_s)` with the family's largest checkpoint as `u`. Floor factors cancel algebraically in this ratio. Absolute/relative selection becomes redundant when headroom is active.
- Sub-floor corrected scores, zero denominators (`1e-12`) and negative headroom are excluded with reason codes; negative raw capability changes remain signed.
- Algebraic equivalence preserves guard domains. Empirically indistinguishable per-axis vectors and eligibility masks are then folded (precision `1e-12`), with structural and observed class counts both recorded. A repeated setting never receives extra weight simply for having another label.
- Ties within `.01` receive fractional credit. Stable-axis sets are intersections of winner sets. If all axes tie everywhere, F/selection is undefined.
- Parse exclusion uses the common valid scoring-unit intersection within the ladder. All five paraphrases must parse and be correct for consistency to pass. Floors are recomputed from the same retained units.
- Primary bootstrap is paired across models and baselines and stratified across the three equal-weight POPE/grounding subsets. Consistency resamples original questions, not paraphrases. Image-cluster bootstrap is a separate sensitivity analysis.
- F_noise fixes absolute drop, no floor, no headroom, mean OWA and failure=wrong. Its sampling CI uses a nested paired bootstrap. The separate Monte-Carlo CI measures precision of the finite bootstrap calculation and is **not** used to open the training gate.

Formal C uses 1,000 outer resamples; its nested F_noise calculation uses 1,000 inner resamples. Raw paired endpoint gap/group-size patterns are sufficient statistics for the fixed mean-OWA noise ruler; multinomial pattern sampling is equivalent to drawing the original units/groups.

## Training data and loss

The user explicitly approved independent task proxies:

- POPE-like object presence: COCO train images and original object annotations, deterministic balanced positive/negative questions.
- Grounding: RefCOCO/+/g train references.
- Consistency: MMBench original questions not selected for evaluation, after image-level separation.
- Reasoning: author-endorsed ScienceQA train, as a **training proxy for MMStar**, never MMStar evaluation items.

Training images are checked against every evaluation image using both byte SHA256 and decoded RGB pixel SHA256. This does not claim to detect arbitrary resized/cropped near duplicates. The frozen pool and provenance record the actual selected counts; 600 units/axis is the target cap, not a claimed count. Consistency retains five template variants per original question.

After a formal C3 go decision, A1 uses the top Qwen-family axis, A2 the runner-up, U a balanced task mixture, and N is a true no-op reference. An unresolved top-axis tie stops selection. A1/A2/U each receive 1,000 optimizer updates (r=16, alpha=32, lr=1e-4); N has the same allotted step budget and zero effective updates. Because N's parameters never change, its deterministic evaluation ledger is the existing base ledger.

Teacher Qwen 7B is unloaded before student Qwen 3B trains. Top-50 logits **and the full-vocabulary log normalizer** are cached for each assistant target position. Distillation uses top-50 plus an explicit omitted-vocabulary tail bucket, temperature 1, mixed 50/50 with teacher-response cross entropy. Teacher/student token IDs and vocabulary mappings must match. LoRA targets language attention q/k/v/o projections, uses zero dropout, gradient checkpointing, batch size 1 and AdamW with zero weight decay. These loss/target-module choices supplement the standard card and are documented, not attributed to its text.

Formal training cannot use a synthetic/smoke gate, a changed ledger or a changed protocol. Post-training inference uses the same frozen evaluation manifest and greedy fp16 settings. Net gain retains signed components; guard-invalid transformed sums remain undefined rather than silently treating missing axes as zero.

## Gate and execution

The pipeline preserves the reference's falsification stop: if every ladder has a stable axis and its scale flip rate is no greater than the noise CI upper bound, record no-go. Go requires separated scale/noise sampling intervals on at least one ladder; other cases are indeterminate and do not start training. This conservative handling of inconclusive results is an explicit implementation decision.

```bash
source /etc/network_turbo
cd /root/autodl-tmp/arp
/root/autodl-tmp/arp-venv/bin/python tools/preflight.py cpu
/root/autodl-tmp/arp-venv/bin/python tools/preflight.py gpu
# After preflight ready.json exists, a detached screen job runs scripts/formal.sh.
```

Preflight exercises real inference on eight inputs per checkpoint, all four tasks, real teacher caching, eight optimizer steps covering all four training axes, and reloaded-adapter inference. AMP overflows reduce the scale and retry the same logical step; skipped updates never count toward the formal 1,000-update budget. Synthetic CPU fixtures are labelled and kept only under `runs/preflight`; their gate always has `formal=false`.

The formal job runs B → C → E analysis exports → conditional D → E final exports. It writes `runs/formal-20261014/status.json` and `console.log`, and resumes per-item inference and optimizer checkpoints. A failed infrastructure operation stops the affected stage; it is never converted into a wrong-answer observation.

## GitHub storage

Source code and protocol are on branch `codex/arp-autodl` in `Jerry-114514/Jerry-vibe-coding`.
Necessary remote outputs are uploaded as GitHub Release assets (compressed archives split below 1 GiB): raw generations, parsed answers, flags, metadata, training manifests and image/provenance inventories, teacher logits once, LoRA adapters, optimizer resume states, losses, bootstrap arrays, tables and figures. No pretrained base weights or SSH/private credentials are published.

The GitHub credential is obtained through the existing local credential helper, transmitted via authenticated SSH stdin, and retained only in the detached job's process environment. No credential is written to the repository or artifact archive. Training outputs travel AutoDL → GitHub; they are not downloaded to the local workspace.

## Official implementation references

- [Qwen model and preprocessing](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [InternVL3 model and tiling](https://huggingface.co/OpenGVLab/InternVL3-1B)
- [LLaVA-OneVision HF interface](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- [ScienceQA author-endorsed data access](https://github.com/lupantech/ScienceQA#-download-the-dataset)
- [AutoDL academic acceleration](https://api.autodl.com/docs/network_turbo/)
- [GitHub large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
