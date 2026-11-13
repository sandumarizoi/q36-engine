# q36

A from-scratch inference engine for **Qwen3.6-35B-A3B** on an **Intel Arc B580** (12 GB), written in C++ with SYCL/ESIMD.
It runs a single model on a single GPU and squeezes everything out of that combination: a per-expert VRAM cache with the
rest of the MoE computed on the CPU, multi-token prediction (MTP) drafts, and a prompt-processing path that runs on the
GPU's XMX matrix engines.

It ships as an OpenAI-compatible HTTP server.

> q36 is not a general-purpose runtime. It supports exactly one model file (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`) and one GPU
> architecture (Xe2 / `bmg-g21`). Weight shapes and quantization types are baked into the kernels.

## Features

- **Per-expert VRAM cache.** Dense weights stay on the GPU; of the 10,240 routed experts the hottest ~37% live in VRAM and
  are swapped in the background as the topic changes. Cache misses are computed on the CPU in parallel with the GPU.
- **MTP speculative decoding** with the model's own MTP head, adaptive draft depth and free state rollback for the
  Gated DeltaNet layers.
- **Fast prompt processing.** Long prompts go through the model in windows of 4096 tokens, layer by layer: int8 GEMM,
  K-quant expert GEMM and flash attention on XMX (`dpas`), with uncached experts streamed over PCIe while the GPU computes.
- **int8 K/V cache** (~13 KB per token), contexts up to 64k tokens.
- **OpenAI-compatible server**: `/v1/chat/completions` (streaming, tool calls, reasoning), `/v1/models`, `/health`.
- Lossless conversion: the engine's model file holds exactly the quantized values of the GGUF, only re-laid out.

## Performance

Test system: Arc B580 12 GB (PCIe 3.0 x8), Ryzen 5 5500, 32 GB DDR4-3600, Windows 11. Server defaults
(`--ctx 32768 --vram-gb 10.5 --mtp 2`), single request.

| workload | tokens/s |
|---|---|
| code generation (greedy / sampled) | 128–133 / 128 |
| Ukrainian prose | 126–136 |
| follow-up turn with reasoning (Ukrainian) | 122 |
| generation after a 16k-token prompt | 94 |
| generation after a 30k-token prompt (`--ctx 65536`) | 83 |
| prompt processing, 3k–16k tokens | 1100–1400 |
| prompt processing, 30k tokens | 1100 |

Generation speed depends on how well the expert cache matches the current text; it is lower for the first few hundred
tokens after a topic switch.

## Requirements

- Windows 10/11 x64
- Intel Arc B580 with a current driver
- 32 GB RAM (the model file is read fully into memory), ~23 GB of disk for the converted model
- Visual Studio 2022 Build Tools (C++ workload)
- Intel oneAPI DPC++/C++ Compiler 2025.x or newer (tested with 2026.2)
- The model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` from Unsloth's Qwen3.6-35B-A3B GGUF release (the file includes the MTP layer)

## Build

```bat
build.cmd
```

This compiles the server, a CLI and the two converters, and copies them together with the oneAPI runtime DLLs into
`bin\`. Building the GPU programs takes a few minutes because the kernels are compiled ahead of time for `bmg-g21`.

If oneAPI or Visual Studio are not in their default locations, set `ONEAPI_ROOT` (folder with `setvars.bat`) and/or
`VSVARS` (path to `vcvars64.bat`) before running `build.cmd`. `SETVARS_ARGS` is passed to `setvars.bat`, e.g.
`--config=oneapi.cfg` to select component versions when several are installed.

## Convert the model

```bat
bin\pack.exe  path\to\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf models\Qwen3.6-35B-A3B-UD-Q4_K_M.q36
bin\mktok.exe path\to\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf models\Qwen3.6-35B-A3B-UD-Q4_K_M.tok
```

The `.tok` file (tokenizer and chat template) must sit next to the `.q36` file under the same name. The GGUF is not
needed afterwards. On the first run the server creates `<model>.q36.prof`, a small file with expert usage statistics
that is used to warm up the cache on the next start.

## Run

```bat
start.cmd
```

`start.cmd` uses `models\Qwen3.6-35B-A3B-UD-Q4_K_M.q36` unless `MODEL` is set, and passes extra arguments through
(`start.cmd --ctx 65536`). The API is then available at `http://127.0.0.1:9936/v1`; any OpenAI-compatible client works:

```bash
curl http://127.0.0.1:9936/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

Loading takes ~2 s when the model file is in the OS file cache, otherwise it is bound by disk speed.

### Server options

| option | default | description |
|---|---|---|
| `--ctx N` | 32768 | context length in tokens (K/V cache: ~13 KB per token) |
| `--vram-gb X` | 10.5 | total VRAM the engine may use; everything left after the model and context goes to the expert cache |
| `--cache-gb X` | | expert cache size, overrides `--vram-gb` |
| `--mtp N` | 2 | maximum MTP draft depth (0 = off, up to 7) |
| `--prefill-window N` | 4096 | window size of the GPU prompt path (0 = always step by step) |
| `--prefill-min N` | 512 | prompts shorter than this are processed step by step |
| `--port N`, `--host A` | 9936, 127.0.0.1 | listen address |
| `--threads N` | 6 | CPU threads for cache misses |
| `--tok F`, `--profile F` | next to the model | tokenizer file, expert usage profile |
| `--draft-vocab N` | 65536 | MTP drafts are computed over the first N vocabulary rows plus all Cyrillic tokens |
| `--draft-vocab-extra F` | | file with additional token ids for drafts (replaces the automatic Cyrillic set) |
| `--lookup 1` | 0 | experimental prompt-lookup drafts |
| `--alias S` | Qwen3.6-35B-A3B-q36 | model name reported by `/v1/models` |
| `--log-level N`, `--log-interval S` | 1, 1 | progress lines during prefill and generation (2 = with a per-token time breakdown) |

### Request fields

`messages`, `tools`, `stream`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`,
`chat_template_kwargs.enable_thinking`, `chat_template_kwargs.preserve_thinking`. Reasoning is returned in
`reasoning_content`, tool calls in `tool_calls`. A llama.cpp-style `/completion` endpoint with raw token ids is also available.

## How it works

- **Model file.** Every quantized matrix is stored row by row (one contiguous record per row); matrices that share an
  input are fused; each expert is one record with its gate/up and down rows. This makes single experts cheap to move
  between RAM and VRAM.
- **Decode.** Each layer runs as two recorded SYCL graphs: attention / Gated DeltaNet plus router, then the cached experts.
  Between them the host computes the missed experts with AVX2 int8 kernels. A background thread uploads experts that became
  hot and evicts cold ones.
- **MTP.** Drafts come from the model's MTP block and are verified in one batched step; rejected positions are simply
  overwritten, because the recurrent state is applied one step late.
- **Prompt processing.** A window of tokens is processed layer by layer. All matrix work uses XMX: Q8_0 GEMM on activation
  tiles, experts grouped by routing on the GPU, flash attention with int8 Q·K and fp16 P·V. Uncached experts are streamed
  through pinned buffers into memory temporarily borrowed from the expert cache. Afterwards the cache is refilled with the
  experts the prompt used most.

## Project layout

| path | contents |
|---|---|
| `src/gpu_kernels.hpp` | decode kernels (SYCL ESIMD) |
| `src/gpu_prefill.hpp` | prompt-processing kernels (XMX GEMM, grouped experts, flash attention) |
| `src/engine.hpp` | engine: decode step, expert cache, MTP, prefill windows, checkpoints |
| `src/pack.hpp`, `src/cpu.hpp` | model file format, CPU expert kernels, thread pool |
| `src/gguf.hpp`, `src/model.hpp` | GGUF reader, model description used by the converter |
| `src/tokenizer.hpp`, `src/chat.hpp`, `src/json.hpp` | tokenizer, chat template, JSON |
| `tools/server.cpp` | HTTP server |
| `tools/q36.cpp` | CLI on token ids (benchmarks, `--kprof`) |
| `tools/pack.cpp`, `tools/mktok.cpp` | GGUF → engine model file, GGUF → tokenizer file |

## Limitations

- One model, one GPU architecture, Windows only.
- One request at a time; text only (no vision); no context shift — a prompt that does not fit is rejected.

## License

MIT, see [LICENSE](LICENSE).
