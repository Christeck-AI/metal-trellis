#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_qwen35moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_35B_A3B; break;
        case 48: type = LLM_TYPE_122B_A10B; break;
        case 60: type = LLM_TYPE_397B_A17B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // Packed EXL3 dense linears (fork metal-exl3, Paso 2c). A pack whose
    // trellis is nullptr means the GGUF carries the stock weight instead
    // (fallback: the same binary loads a Q8_0 GGUF).
    trellis.resize(n_layer);
    auto make_pack = [&](llm_tensor tensor, int il, int64_t k, int64_t n, int K, int cb = 1) {
        trellis_pack pack;
        pack.K  = K;
        pack.cb = cb;
        const int64_t n_elem = (k / 16) * (n / 16) * 16 * K;
        // input/output layer tensors (lm_head) carry no layer number:
        // create_tensor aborts unless tn.bid == -1 for them.
        const int bid = (tensor == LLM_TENSOR_OUTPUT) ? -1 : il;
        pack.trellis = create_tensor(tn(tensor, "trellis", bid), { n_elem }, TENSOR_NOT_REQUIRED);
        if (!pack.trellis) {
            return pack;
        }
        pack.suh = create_tensor(tn(tensor, "suh", bid), { k }, TENSOR_NOT_REQUIRED);
        pack.svh = create_tensor(tn(tensor, "svh", bid), { n }, TENSOR_NOT_REQUIRED);
        return pack;
    };

    // stacked expert packs (Paso 3): trellis I16 flat [n_exp*tk*tn*16*K],
    // suh F32 [n_exp*k], svh F32 [n_exp*n]; expert-major order
    auto make_pack_exps = [&](llm_tensor tensor, int il, int64_t k, int64_t n, int64_t n_exp, int K, int cb = 1) {
        trellis_pack pack;
        pack.K  = K;
        pack.cb = cb;
        const int64_t n_elem = n_exp * (k / 16) * (n / 16) * 16 * K;
        pack.trellis = create_tensor(tn(tensor, "trellis", il), { n_elem }, TENSOR_NOT_REQUIRED);
        if (!pack.trellis) {
            return pack;
        }
        pack.suh = create_tensor(tn(tensor, "suh", il), { n_exp * k }, TENSOR_NOT_REQUIRED);
        pack.svh = create_tensor(tn(tensor, "svh", il), { n_exp * n }, TENSOR_NOT_REQUIRED);
        return pack;
    };

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output (lm_head: packed trellis, K=6; fallback to the stock weight)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output_pack = make_pack(LLM_TENSOR_OUTPUT, 0, n_embd, n_vocab, 6);
    if (!output_pack.trellis) {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    }

    // if output is NULL, init from the input tok embed
    if (output == NULL && !output_pack.trellis) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers (packed trellis, K=3; fallback to stock Q/K/V)
            trellis[il].wq = make_pack(LLM_TENSOR_ATTN_Q, il, n_embd, n_embd_head_k * n_head * 2, 3);
            trellis[il].wk = make_pack(LLM_TENSOR_ATTN_K, il, n_embd, n_embd_k_gqa, 3);
            trellis[il].wv = make_pack(LLM_TENSOR_ATTN_V, il, n_embd, n_embd_v_gqa, 3);
            if (!trellis[il].wq.trellis) {
                create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            }
            trellis[il].wo = make_pack(LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head, n_embd, 3);
            if (!trellis[il].wo.trellis) {
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);
            }

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            trellis[il].wqkv = make_pack(LLM_TENSOR_ATTN_QKV, il, n_embd, key_dim * 2 + value_dim, 3);
            if (!trellis[il].wqkv.trellis) {
                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            }
            trellis[il].wqkv_gate = make_pack(LLM_TENSOR_ATTN_GATE, il, n_embd, value_dim, 3);
            if (!trellis[il].wqkv_gate.trellis) {
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            }
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            trellis[il].ssm_out = make_pack(LLM_TENSOR_SSM_OUT, il, value_dim, n_embd, 3);
            if (!trellis[il].ssm_out.trellis) {
                layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", il), { value_dim, n_embd }, flags);
            }
        }

        // Routed experts (packed EXL3 trellis when present, Paso 3;
        // fallback to the stock stacked weights)
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        trellis[il].ffn_gate_exps = make_pack_exps(LLM_TENSOR_FFN_GATE_EXPS, il, n_embd, n_ff_exp, n_expert, 3);
        trellis[il].ffn_up_exps   = make_pack_exps(LLM_TENSOR_FFN_UP_EXPS,   il, n_embd, n_ff_exp, n_expert, 3);
        trellis[il].ffn_down_exps = make_pack_exps(LLM_TENSOR_FFN_DOWN_EXPS, il, n_ff_exp, n_embd, n_expert, 3);
        if (!trellis[il].ffn_gate_exps.trellis) {
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        }

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        trellis[il].ffn_gate_shexp = make_pack(LLM_TENSOR_FFN_GATE_SHEXP, il, n_embd, n_ff_shexp, 3);
        if (!trellis[il].ffn_gate_shexp.trellis) {
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, n_ff_shexp }, flags);
        }
        trellis[il].ffn_up_shexp = make_pack(LLM_TENSOR_FFN_UP_SHEXP, il, n_embd, n_ff_shexp, 3);
        if (!trellis[il].ffn_up_shexp.trellis) {
            layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, n_ff_shexp }, flags);
        }
        trellis[il].ffn_down_shexp = make_pack(LLM_TENSOR_FFN_DOWN_SHEXP, il, n_ff_shexp, n_embd, 3);
        if (!trellis[il].ffn_down_shexp.trellis) {
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_shexp, n_embd }, flags);
        }
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, mtp_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, mtp_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, mtp_flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, mtp_flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_qwen35moe::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // post-norm hidden state feeds both the LM head and the MTP seed below
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head (packed trellis lm_head, K=6; fallback to the stock weight)
    const auto & output_pack = static_cast<const llama_model_qwen35moe &>(model).output_pack;
    cur = output_pack.trellis
        ? build_trellis_mm(output_pack, cur)
        : build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

const llama_model_qwen35moe::trellis_layer_packs & llama_model_qwen35moe::graph::trellis_packs(int il) const {
    return static_cast<const llama_model_qwen35moe &>(model).trellis[il];
}

ggml_tensor * llama_model_qwen35moe::graph::build_trellis_mm(
        const llama_model_qwen35moe::trellis_pack & pack, ggml_tensor * cur) {
    // EXL3 trellis linear: y = svh * H128(H128(x * suh) @ W_inner).
    // One token runs the v3 GEMV_NB chain; prefill runs the multi-row
    // GEMM_NB (DISENO §12), which decodes each weight once per 16-row
    // block without materializing W_inner.
    const int64_t k = cur->ne[0];
    const int64_t n = pack.svh->ne[0];
    // token count from the actual input: the last layer crops cur to the
    // output tokens (ggml_get_rows, crop_before_nextn), so cur can carry
    // 1 token while the graph's n_tokens is 512. (Paso 2c fix.)
    const int64_t S = ggml_nelements(cur) / k;
    if (S == 1) {
        // v3 path (Paso 2e v3): MUL(suh) -> FWHT -> GEMV_NB
        // (barrier-free GEMV, 16-col tiles like the MLX opt4)
        // -> FWHT_SVH. Four nodes per linear instead of six;
        // the v1/v2 single-kernel fusion was correct but slow
        // (128 cols per threadgroup starved the GPU).
        ggml_tensor * x = ggml_mul(ctx0, cur, pack.suh);
        ggml_tensor * t = ggml_reshape_2d(ctx0, x, 128, k / 128);
        t = ggml_trellis_fwht(ctx0, t);
        t = ggml_reshape_1d(ctx0, t, k);
        t = ggml_trellis_gemv_nb(ctx0, pack.trellis, t, pack.K, pack.cb);
        t = ggml_trellis_fwht_svh(ctx0, t, pack.svh);
        return ggml_reshape_2d(ctx0, t, n, S);
    }
    ggml_tensor * x = ggml_mul(ctx0, cur, pack.suh);
    ggml_tensor * xh = ggml_reshape_3d(ctx0, x, 128, k / 128, S);
    xh = ggml_trellis_fwht(ctx0, xh);
    xh = ggml_reshape_2d(ctx0, xh, k, S);
    ggml_tensor * y = ggml_trellis_gemm_nb(ctx0, pack.trellis, xh, pack.K, pack.cb);
    y = ggml_reshape_3d(ctx0, y, 128, n / 128, S);
    y = ggml_trellis_fwht(ctx0, y);
    y = ggml_reshape_2d(ctx0, y, n, S);
    y = ggml_mul(ctx0, y, pack.svh);
    return y;
}

ggml_tensor * llama_model_qwen35moe::graph::build_mm_or_trellis(
        const llama_model_qwen35moe::trellis_pack & pack,
        ggml_tensor * w, ggml_tensor * w_s, ggml_tensor * cur) {
    if (pack.trellis) {
        return build_trellis_mm(pack, cur);
    }
    return build_lora_mm(w, cur, w_s);
}

ggml_tensor * llama_model_qwen35moe::graph::build_trellis_mm_id(
        const llama_model_qwen35moe::trellis_pack & pack, ggml_tensor * cur, ggml_tensor * ids) {
    // EXL3 trellis expert linear: y = svh[e] * H128(H128(x * suh[e]) @ W_inner[e])
    // per (token, slot), e = ids[slot, token]. cur is [k, tokens] (shared
    // input, gate/up) or [k, topk, tokens] (per-slot input, down).
    const int64_t n = ggml_nelements(pack.svh) / n_expert;
    const int64_t k = cur->ne[0];
    const int64_t tokens = ids->ne[1];
    // ids comes from ggml_argsort_top_k, a view with the full argsort
    // stride; the trellis ID ops require contiguous inputs
    cur = ggml_cont(ctx0, cur);
    ids = ggml_cont(ctx0, ids);
    ggml_tensor * xh = ggml_trellis_xh_id(ctx0, pack.suh, cur, ids);
    ggml_tensor * c;
    if (tokens >= 64) {
        // grouped prefill (DISENO §12): decode every expert once to
        // fp16 scratch and let the stock mul_mat_id group the rows.
        // Below the threshold the per-(token, slot) GEMV chain wins:
        // decoding all 256 experts is a fixed cost per prefill.
        ggml_tensor * w = ggml_trellis_decode_id(ctx0, pack.trellis, k, n, n_expert, pack.K, pack.cb);
        c = ggml_mul_mat_id(ctx0, w, xh, ids);
    } else {
        c = ggml_trellis_gemv_nb_id(ctx0, pack.trellis, xh, ids, n, pack.K, pack.cb);
    }
    return ggml_trellis_fwht_svh_id(ctx0, c, pack.svh, ids);
}

ggml_tensor * llama_model_qwen35moe::graph::build_moe_ffn_trellis(ggml_tensor * cur, const int il) {
    // Mirror of build_moe_ffn for this arch (SOFTMAX gating, norm_w, SILU)
    // with the expert linears as trellis ops (Paso 3). No MoE expert
    // cache: the packed weights are already compact (3 bits).
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const auto & layer = model.layers[il];
    const auto & packs = trellis_packs(il);

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);
    ggml_tensor * probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
    cb(probs, "ffn_moe_probs", il);

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    // norm_w, same as stock
    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
    cb(weights_sum, "ffn_moe_weights_sum", il);
    weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
    weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights_norm", il);
    weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    ggml_build_forward_expand(gf, weights);
    ggml_build_forward_expand(gf, cur);

    ggml_tensor * gate = build_trellis_mm_id(packs.ffn_gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(gate, "ffn_moe_gate", il);
    ggml_tensor * up = build_trellis_mm_id(packs.ffn_up_exps, cur, selected_experts);
    cb(up, "ffn_moe_up", il);

    ggml_tensor * act;
    const float limit = hparams.swiglu_clamp_exp[il];
    if (limit > 1e-6f) {
        up = ggml_clamp(ctx0, up, -limit, limit);
        ggml_tensor * gate_act = ggml_silu(ctx0, gate);
        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
        act = ggml_mul(ctx0, gate_act, up);
        cb(act, "ffn_moe_swiglu_limited", il);
    } else {
        act = ggml_swiglu_split(ctx0, gate, up);
        cb(act, "ffn_moe_swiglu", il);
    }

    ggml_tensor * experts = build_trellis_mm_id(packs.ffn_down_exps, act, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    ggml_tensor * moe_out = cur_experts[0];
    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }
    if (n_expert_used_il == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }
    cb(moe_out, "ffn_moe_out", il);
    return moe_out;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35moe::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_mm_or_trellis(trellis_packs(il).wqkv, model.layers[il].wqkv, model.layers[il].wqkv_s, input);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_mm_or_trellis(trellis_packs(il).wqkv_gate, model.layers[il].wqkv_gate, model.layers[il].wqkv_gate_s, input);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full;
    ggml_tensor * Kcur;
    ggml_tensor * Vcur;
    if (trellis_packs(il).wq.trellis) {
        // packed EXL3 Q/K/V (Paso 2c)
        Qcur_full = build_trellis_mm(trellis_packs(il).wq, cur);
        Kcur      = build_trellis_mm(trellis_packs(il).wk, cur);
        Vcur      = build_trellis_mm(trellis_packs(il).wv, cur);
    } else {
        const auto qkv = build_qkv(model.layers[il], cur,
                n_embd_head * 2, n_head,
                n_embd_head,     n_head_kv,
                n_embd_head,     n_head_kv,
                il, false);
        Qcur_full = qkv.q;
        Kcur      = qkv.k;
        Vcur      = qkv.v;
    }
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_mm_or_trellis(trellis_packs(il).wo, model.layers[il].wo, model.layers[il].wo_s, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);

    // Output projection
    cur = build_mm_or_trellis(trellis_packs(il).ssm_out, model.layers[il].ssm_out, model.layers[il].ssm_out_s, final_output);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out = trellis_packs(il).ffn_gate_exps.trellis
        ? build_moe_ffn_trellis(cur, il)
        : build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr || trellis_packs(il).ffn_up_shexp.trellis) {
        ggml_tensor * ffn_shexp;
        if (trellis_packs(il).ffn_up_shexp.trellis) {
            // packed EXL3 shared expert (Paso 2c): SwiGLU, same as build_ffn
            // (swiglu_clamp_shexp is 0 for Qwen3.5, so no clamp path)
            ggml_tensor * gate = build_trellis_mm(trellis_packs(il).ffn_gate_shexp, cur);
            ggml_tensor * up   = build_trellis_mm(trellis_packs(il).ffn_up_shexp, cur);
            ggml_tensor * act  = ggml_swiglu_split(ctx0, gate, up);
            ffn_shexp = build_trellis_mm(trellis_packs(il).ffn_down_shexp, act);
        } else {
            ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                    model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                    model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35MOE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "MTP block missing ffn_gate_inp");

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "mtp_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "mtp_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn= cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
