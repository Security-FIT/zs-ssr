# SGE job scripts

The job scripts that produced the published results, used in our computation cluster.

- `freevc/`: FreeVC on synthesizers 1..8, and the aligned (`--align`) runs on 1 and 4.
- `againvc/`, `diffvc/`, `wav2vecvc/`: AGAIN-VC, DiffVC and wav2vec-VC on synthesizers 1..8.
  The wav2vec-VC code itself is not published (see the main README).
- `analysis/`: the paper analysis (`analysis/*.py`): VC quality and naturalness, reconstruction
  naturalness, threshold impact.

Every script sources `_sge_header.sh`, which is not published. It only held environment setup
specific to our infrastructure: shell strict mode and a CPU-time limit, conda activation of the
`zero-shot-ssr` env, `CUDA_HOME`, picking a free GPU on the node, offline paths for the Coqui,
HuggingFace and torch.hub caches, and `cd` into the repository (`$SRCDIR`). Replace line
with the equivalent setup for your cluster.

