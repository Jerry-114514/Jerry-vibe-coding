# Independent route B: dataset delivery audit

Verified on 2026-10-16. This is a data delivery and engineering verification, not an ARP stage-C decision or a formal evaluation result.

The current 3,545-row training pool references 2,132 images. The 16,900-prompt evaluation manifest references 4,705 images. All 6,837 images and six manifests/locks (6,843 files total) match the historical archive inventory by actual file size and SHA256. The historical inventory and archive also match the current GitHub server asset SHA256 digests. No large payload was downloaded locally.

- [Machine-readable audit](route-b-dataset-delivery-verification.json)
- [Historical dataset release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-formal-20261014)
- [Current route-B cache, checkpoints and reports release](https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/arp-protocol-v2-20261016)

Historical archive: `arp-formal-20261014-20261016T004018Z.tar.gz`, 1,064,822,655 bytes, SHA256 `fcf9780dab9d9533072851e9ee2aa567d6926431e611d75d62d4700d9ab4ec68`. Inventory: `arp-formal-20261014-20261016T004018Z-inventory.json`, SHA256 `a44bf815973ea41d3efa9ca3f88642b68e9336cd09d75eab39b427a0300cdf3f`.

The newer EOS teacher cache has its own delivery record. Base model weights are independently pinned and verified; this dataset audit does not claim to archive those weights.

At the 23:51 Beijing runtime check, nine of fifteen formal training runs had completed. All nine have also passed independent verification of 1,000 contiguous real updates, finite dual CE, nonzero gradients, actual optimizer step 1,000, and exact tensor equality between exported and recovery-state adapters. Seeds 42, 137 and 256 are covered for A1/A2/U. The remaining six runs and full evaluation are still pending. No claim of complete experiment results is made here.
