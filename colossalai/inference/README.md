# 🚀 Colossal-Inference

## Table of contents

## Introduction

`Colossal Inference` is a module that contains colossal-ai designed inference framework, featuring high performance, steady and easy usability. `Colossal Inference` incorporated the advantages of the latest open-source inference systems, including TGI, vLLM, FasterTransformer, LightLLM and flash attention. while combining the design of Colossal AI, especially Shardformer, to reduce the learning curve for users.

## Design

Colossal Inference is composed of two main components:

1. High performance kernels and ops: which are inspired from existing libraries and modified correspondingly.
2. Efficient memory management mechanism：which includes the key-value cache manager, allowing for zero memory waste during inference.
   1. `cache manager`: serves as a memory manager to help manage the key-value cache, it integrates functions such as memory allocation, indexing and release.
   2. `batch_infer_info`: holds all essential elements of a batch inference, which is updated every batch.
3. High-level inference engine combined with `Shardformer`: it allows our inference framework to easily invoke and utilize various parallel methods.
   1. `engine.TPInferEngine`: it is a high level interface that integrates with shardformer, especially for multi-card (tensor parallel) inference:
   2. `modeling.llama.LlamaInferenceForwards`: contains the `forward` methods for llama inference. (in this case : llama)
   3. `policies.llama.LlamaModelInferPolicy` : contains the policies for `llama` models, which is used to call `shardformer` and segmentate the model forward in tensor parallelism way.

## Pipeline of inference:

In this section we discuss how the colossal inference works and integrates with the `Shardformer` . The details can be found in our codes.

![Colossal-Inference](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Colossal-inference.png)

## Roadmap of our implementation

- [x] Design cache manager and batch infer state
- [x] Design TpInference engine to integrates with `Shardformer`
- [x] Register corresponding high-performance `kernel` and `ops`
- [x] Design policies and forwards (e.g. `Llama` and `Bloom`)
  - [x] policy
  - [x] context forward
  - [x] token forward
- [ ] Replace the kernels with `faster-transformer` in token-forward stage
- [ ] Support all models
  - [x] Llama
  - [x] Bloom
  - [ ] Chatglm2
- [ ] Benchmarking for all models

## Get started

### Installation

```bash
pip install -e .
```

### Requirements

dependencies

```bash
pytorch= 1.13.1 (gpu)
transformers= 4.30.2
triton==2.0.0.dev20221202
vllm=
flash-attention=
```

### Docker

You can use our official docker container as well.

```bash
docker..
```

### Dive into fast-inference!

example files are in

```bash
cd colossalai.examples
python xx
```

## Performance

### environment:

We conducted [benchmark tests](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/performance_benchmark.py) to evaluate the performance. We compared the inference `latency` and `throughputs` between `colossal-inference` and `torch`.

We set the batch size to 4, the number of attention heads to 8, and the head dimension to 64. `N_CTX` refers to the sequence length.

In the case of using 2 GPUs, the results are as follows.

###