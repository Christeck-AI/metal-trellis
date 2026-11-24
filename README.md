# Metal Trellis

**EXL3 (trellis-quantized) inference on Apple Silicon, via Metal.**

Metal Trellis is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs
[EXL3](https://github.com/turboderp-org/exllamav3) checkpoints — trellis-coded weights at
~3 bits per weight — on Apple Silicon GPUs. The trellis weights are decoded on the fly
inside ggml Metal kernels; the model never needs to be dequantized into memory.

Verified targets: **Qwen3.5-35B-A3B** (MoE) and **Qwen3.8-27B** (dense), quantized with
ExLlamaV3 by Turboderp at 3.00 bpw.

> Framing, honestly: the **MoE path is the practical one**. The dense path works and is
> token-identical to its reference, but it is slow — it exists as research and as a
> correctness proof. **The engine is the product.**

## What was added to llama.cpp

- **EXL3 safetensors reader** (`src/llama-safetensors.*`) — stock llama.cpp has none.
- **Trellis decode** (`src/llama-exl3-decode.*`) — all three EXL3 codebooks, byte-verified
  against real checkpoints.
- **ggml trellis ops** (`GGML_OP_TRELLIS_*`): `MUL_MAT`, `FWHT`, `DECODE`,
  `MUL_MAT_FUSED`, `GEMV_NB`, `FWHT_SVH`, `XH_ID`, `GEMV_NB_ID`, `FWHT_SVH_ID`,
  `DECODE_ID` — Metal kernels in `ggml/src/ggml-metal/kernels/trellis.metal`,
  with CPU reference paths.
- **Packed GGUF route** — dense linears and stacked MoE experts packed at 3.0 bpw,
  produced by the conversion tooling in `conversion/` (EXL3 handling vendored from
  the trellis-serve reference, see NOTICE).
- **Model wiring** for `qwen35moe` (35B MoE) and `qwen35` (27B dense, K 2..5, mul1),
  with stock fallback for non-trellis tensors.

## Build

Requirements: macOS on Apple Silicon, Xcode Command Line Tools, CMake ≥ 3.18.
Node/npm are needed only if you build the embedded Web UI from source (see note).

```sh
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binaries land in `build/bin/` (`llama-cli`, `llama-server`, …).

> **Web UI assets note:** `llama-server` embeds a Web UI. By default the build tries to
> download prebuilt UI assets from the ggml-org HF bucket. If that download fails or the
> bundle doesn't validate, build the UI from source instead (requires npm):
>
> ```sh
> cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=ON
> ```

## Convert and run

1. **Get an EXL3 checkpoint.** Either quantize a BF16 model with
   [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (CUDA), or download one of
   Turboderp's published EXL3 packs from Hugging Face (e.g. Qwen3.5-35B-A3B at 3.00 bpw).
2. **Convert to Packed GGUF** with this repo's converter (it understands EXL3 tensors):

   ```sh
   python convert_hf_to_gguf.py <exl3-checkpoint-dir> --outfile model-packed.gguf
   ```

3. **Run** as usual — the Metal backend decodes trellis weights on the fly:

   ```sh
   ./build/bin/llama-cli -m model-packed.gguf -p "Hello" -ngl 99
   ./build/bin/llama-server -m model-packed.gguf -ngl 99 --port 8080
   ```

## Measurements (M1 Max, 64 GB)

All numbers measured on the fork; methodology and raw actas live with the project.

| Check | Result |
|---|---|
| Quality floor (27B): EXL3 3.00 bpw vs BF16 | ΔNLL indistinguishable (confidence interval includes 0) |
| MoE 35B experts decode | 18.7 tok/s |
| MoE 35B prefill TTFT | 0.44 s |
| Packed decode (fused GEMV v3) | 22.2 tok/s (from 13.9 baseline, +60%) |
| Dense 27B | 2.70 tok/s, **100% token-identical** vs Q8_0 reference — the win is memory (11.6 GB packed), not speed |
| Black-hole challenge | First trellis-rendered black hole on Apple Silicon (2026-11-21): geodesic integration, Doppler beaming, gravitational redshift — skeleton quality, works |
| Code bench (CT-Bench-mini full, 60 tasks, hidden tests) | **75% (45/60)** — Qwen3.5-35B-A3B-PACKED2 via Metal Trellis, thinking on, seed 42, 61.2 min, 16.6 tok/s. TypeScript 31/38 · SQL 5/5 · C++ 5/7 · Java 2/4 · JavaScript 2/6 |
| Code bench control — same bench on the unquantized base (Q8_0) | **73% (44/60)** — Qwen3.5-35B-A3B base Q8_0 via llama-metal fork, thinking on, seed 42, 8.6 min, 51.8 tok/s. C++ 7/7 · Java 3/4 · JS 1/6 · SQL 3/5 · TS 30/38. Delta vs PACKED2: −1 task (technical tie) |

> CT-Bench-mini is Christeck's internal coding bench (hidden tests, product-only); scores are published, tasks are not. This run used thinking without a reasoning budget: 2 of 15 failures exhausted the 16384-token cap without delivering code.

## Limitations

- **No encoder yet.** Quantizing a model to trellis still requires ExLlamaV3 on CUDA.
  A native 2–3 bit trellis encoder for the Mac is the roadmap's main item.
- **Dense is slow.** 2.7 tok/s is research territory. Use the MoE for real work.
- **Vision is not wired.** Qwen3.8 is a native VLM, but this fork only verified the
  text path. Wiring the vision tower is open for whoever wants it — instructions and
  measurements are here; bring your own compute.

## Roadmap

- Native trellis encoder (BF16 → 2–3 bpw EXL3) on Apple Silicon, gated by NLL parity
  against CUDA-made checkpoints.
- Vision tower for the dense VLM path.
- Prefill improvements for the dense route.

## License and attribution

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Based on llama.cpp/ggml (MIT,
The ggml authors). EXL3 format and quantization tooling by Turboderp (ExLlamaV3, MIT).
Trellis quantization originates in the QTIP research line. The EXL3 conversion
reference in `conversion/exl3.py` is vendored from trellis-serve.

Model weights are **not** included in this repository. The base models belong to
their authors (Qwen/Alibaba, Apache 2.0); EXL3 packs are Turboderp's work under the
base models' licenses. Convert them yourself with the recipe above, or use the
ready-made Packed GGUFs published by Christeck on Hugging Face.

---

## En español

Metal Trellis corre checkpoints EXL3 (pesos trellis a ~3 bits por peso) en Apple
Silicon usando Metal, decodificando los pesos al vuelo dentro de los kernels de ggml.
Es un fork de llama.cpp. El camino práctico es el MoE (Qwen3.5-35B-A3B, 18.7 tok/s en
expertos); el denso (Qwen3.8-27B) es investigación: correcto token a token, pero lento.
El producto es el motor. La cuantización todavía se hace con ExLlamaV3 en CUDA; el
codificador nativo para Mac es el siguiente paso del roadmap. Licencia MIT, con
atribución a llama.cpp/ggml, Turboderp/ExLlamaV3, QTIP y trellis-serve.

En el bench interno de código CT-Bench-mini (60 tareas, pruebas ocultas), el MoE PACKED2 vía Metal Trellis marcó 75% (45/60) en la M1 Max, con pensamiento activado, semilla 42, 61.2 minutos a 16.6 tok/s. La base sin cuantizar (Q8_0) marcó 73% (44/60) en 8.6 minutos a 51.8 tok/s: un empate técnico.
