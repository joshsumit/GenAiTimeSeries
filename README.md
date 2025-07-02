
This repo contains novel SOTA time series model architectures and enhanced low bit LLM architectures. 

**Temporal Context Embeddings: A Modular Framework for Adaptive and Uncertainty-Aware Time Series Representation** 
  TCE is a modular embedding framework designed   to enhance the adaptability and robustness of time series models. 
  TCE comprises two novel components:
  **Dynamic Patch Morphing (DPM)**, which adaptively reshapes input patches based on local temporal
  structure and **Uncertainty-Aware Embedding (UAE)**, which modulates token representations based
  on predictive uncertainty. Unlike conventional static embeddings, TCE dynamically conditions on both
  temporal context and entropy, enabling improved generalization under noise, distribution shifts, and
  variable-length sequences. TCE is model-agnostic and integrates seamlessly with both **Transformer
  and state-space architectures**. Empirical results across standard time series benchmarks demonstrate
  consistent improvements in accuracy, calibration, and robustness, highlighting TCE’s potential as a
  plug-and-play enhancement for modern time series models.


**DeepTimeNet: A Modular Hybrid Architecture for Robust and Efficient Time Series Modeling**, 
DeepTimeNet is a modular architecture for time series modeling that integrates a **Mamba-
Transformer hybrid backbone** with a **quantized Butterfly decoder**. Designed for robustness, scalability,
and efficiency, DeepTimeNet captures both long-range dependencies and local dynamics while enabling
fast, low-memory inference. While this paper focuses on the backbone and decoder, DeepTimeNet is
compatible with external embedding modules such as Token-Conditioned Embedding (TCE). Extensive
experiments on real-world and synthetic benchmarks demonstrate that DeepTimeNet achieves state-ofthe-
art performance with significantly reduced computational overhead.


**EcoLM: A Log-Quantized, Butterfly-Structured Low-Bit Transformer Language Model**
EcoLM is a highly efficient Transformer-based language model designed for extreme quantization and structured computation. It leverages a log-domain quantization scheme for both activations and weights, enabling sub-2-bit precision without significant performance degradation. The model architecture replaces standard linear layers with ternary-quantized Butterfly Linear Layers, which drastically reduce memory and compute requirements while preserving expressive capacity. EcoLM integrates residual scaling, per-layer gradient clipping, and grouped attention heads to stabilize training under low-bit constraints. This design makes EcoLM particularly suitable for deployment in resource-constrained environments, such as edge devices and low-power inference scenarios. Preliminary results demonstrate competitive performance on standard language modeling benchmarks, with significant gains in efficiency and model compactness.
