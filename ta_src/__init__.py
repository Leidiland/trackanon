# Preload nvidia libs so onnxruntime-gpu finds cuDNN/cuBLAS without LD_LIBRARY_PATH.
from ta_src.utils import onnx_cuda_bootstrap as _ocb  # noqa: F401
