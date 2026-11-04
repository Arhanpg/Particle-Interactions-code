# Third-party notices and source provenance

## Particle-interaction CUDA/HIP source

The imported CUDA and HIP source files come from the public repository:

- Upstream mirror: https://github.com/berenger-eu/gpu-particle-interaction
- Imported revision: `479a812b8cba168f911474f1b04e4c02665aa858`
- Original project identified by the paper: https://gitlab.inria.fr/bramas/gpu-particle-interaction

The upstream README requests that users cite the paper when using, updating, or benchmarking the code. Individual source headers and comments have been preserved verbatim.

### Licensing limitation

The inspected upstream GitHub mirror tree at the imported revision does not contain a top-level `LICENSE` or `LICENCE` file. Some files refer to MIT licensing and the `SpTimer.hpp` header refers to LGPL licensing, but the referenced license texts are not present in that mirror tree.

The destination repository already had a top-level `LICENSE` file before this import. Its presence must **not** be interpreted as automatically changing or governing the license of imported upstream files. Do not remove or overwrite upstream notices. Before redistributing modified copies or asserting a single license for the whole repository, obtain/verify the applicable license texts and file-level terms from the upstream authors or original project.

## Research paper

David Algis, Bérenger Bramas, Emmanuelle Darles, and Lilian Aveneau, “Efficient GPU Implementation of Particle Interactions with Cutoff Radius and Few Particles per Cell,” arXiv:2406.16091v1, 2024. https://arxiv.org/abs/2406.16091
