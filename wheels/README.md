# ViM Wheels

Upload these files under `wheels/` in the Hugging Face model repository:

| Filename | SHA-256 |
|---|---|
| `causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` | `f572458de281ed15fbd6f9262d44d66591416c7a903c4da699ed4d70b3cf5a83` |
| `mamba_ssm-1.2.0.post1+vimcu118torch2.1cxx11abifalse-cp310-cp310-linux_x86_64.whl` | `2906d8cdd3f8f84c91f876f9ddd9337ffbc9161a9e1e66897313438d125b392a` |

The `+vim` wheel is the CUDA 11.8 / PyTorch 2.1 compatibility build used for
paper reproduction and validated against the vendored Vim model. The stock
State Spaces wheel has the same base version but lacks `bimamba_type`.
