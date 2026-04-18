#include "transformer_infer.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "model_weights_generated.h"

namespace {

inline float gelu(float x) {
  return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void layer_norm(const float* x, const float* gamma, const float* beta, int n, float* y) {
  float mean = 0.f;
  for (int i = 0; i < n; ++i) mean += x[i];
  mean /= static_cast<float>(n);
  float var = 0.f;
  for (int i = 0; i < n; ++i) {
    float d = x[i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(n);
  float inv = 1.f / sqrtf(var + mw::kLnEps);
  for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

void linear(const float* x, int in_dim, const float* w_rowmajor, const float* b, int out_dim, float* y) {
  for (int o = 0; o < out_dim; ++o) {
    const float* __restrict__ row = w_rowmajor + o * in_dim;
    float s = b ? b[o] : 0.f;
#pragma GCC unroll 8
    for (int i = 0; i < in_dim; ++i) {
      s += x[i] * row[i];
    }
    y[o] = s;
  }
}

static float g_x[23 * 64];
static float g_tmp[23 * 64];
static float g_qkv[23 * 192];
static float g_attn[23 * 64];
static float g_ff[23 * 256];

void run_encoder_layer(
 int L,
    const float* norm1_w,
    const float* norm1_b,
    const float* in_w,
    const float* in_b,
    const float* out_w,
    const float* out_b,
    const float* norm2_w,
    const float* norm2_b,
    const float* l1_w,
    const float* l1_b,
    const float* l2_w,
    const float* l2_b) {
  const int D = mw::kDModel;
  const int H = mw::kNumHeads;
  const int Dh = D / H;
  const int THREE_D = 3 * D;
  const float inv_sqrt_dh = 1.f / sqrtf(static_cast<float>(Dh));

  for (int l = 0; l < L; ++l) layer_norm(g_x + l * D, norm1_w, norm1_b, D, g_tmp + l * D);

  for (int l = 0; l < L; ++l)
    linear(g_tmp + l * D, D, in_w, in_b, THREE_D, g_qkv + l * THREE_D);

  memset(g_attn, 0, sizeof(float) * static_cast<size_t>(L * D));

  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < L; ++i) {
      float scores[23];
      float maxv = -1e30f;
      for (int j = 0; j < L; ++j) {
        float s = 0.f;
#pragma GCC unroll 16
        for (int d = 0; d < Dh; ++d) {
          const float qv = g_qkv[i * THREE_D + h * Dh + d];
          const float kv = g_qkv[j * THREE_D + D + h * Dh + d];
          s += qv * kv;
        }
        s *= inv_sqrt_dh;
        if (j > i) s = mw::kAttnMask;
        scores[j] = s;
        if (s > maxv) maxv = s;
      }
      float sum = 0.f;
      for (int j = 0; j < L; ++j) {
        scores[j] = expf(scores[j] - maxv);
        sum += scores[j];
      }
      for (int j = 0; j < L; ++j) scores[j] /= sum;

      for (int d = 0; d < Dh; ++d) {
        float acc = 0.f;
        for (int j = 0; j < L; ++j) acc += scores[j] * g_qkv[j * THREE_D + 2 * D + h * Dh + d];
        g_attn[i * D + h * Dh + d] = acc;
      }
    }
  }

  for (int l = 0; l < L; ++l) {
    float outv[64];
    linear(g_attn + l * D, D, out_w, out_b, D, outv);
    for (int i = 0; i < D; ++i) g_x[l * D + i] += outv[i];
  }

  for (int l = 0; l < L; ++l) layer_norm(g_x + l * D, norm2_w, norm2_b, D, g_tmp + l * D);

  for (int l = 0; l < L; ++l) {
    linear(g_tmp + l * D, D, l1_w, l1_b, mw::kFf, g_ff + l * mw::kFf);
    for (int j = 0; j < mw::kFf; ++j) g_ff[l * mw::kFf + j] = gelu(g_ff[l * mw::kFf + j]);
 }

  for (int l = 0; l < L; ++l) {
    float outv[64];
    linear(g_ff + l * mw::kFf, mw::kFf, l2_w, l2_b, D, outv);
    for (int i = 0; i < D; ++i) g_x[l * D + i] += outv[i];
  }
}

}  // namespace

extern "C" int tr_char_to_id(char c) {
  char cl = c;
  if (cl >= 'A' && cl <= 'Z') cl = static_cast<char>(cl - 'A' + 'a');
  for (int i = 0; i < mw::kVocabSize; ++i) {
    if (mw::kVocabChars[i] == cl) return i;
  }
  return mw::kSpaceId;
}

extern "C" void tr_decode_skip_special(const int* ids, int n_ids, char* buf, size_t buf_sz) {
  size_t p = 0;
  for (int i = 0; i < n_ids && p + 1 < buf_sz; ++i) {
    int id = ids[i];
    if (id == mw::kBosId || id == mw::kEosId || id == mw::kPadId) continue;
    buf[p++] = mw::kVocabChars[id];
  }
  buf[p] = '\0';
}

extern "C" void tr_forward_logits_last(const int* tokens, int seq_len, float* logits_out) {
  const int D = mw::kDModel;
  const int V = mw::kVocabSize;
  const int T = mw::kSeqLen;
  if (seq_len <= 0 || seq_len > T) {
    for (int i = 0; i < V; ++i) logits_out[i] = 0.f;
    return;
  }

  for (int l = 0; l < seq_len; ++l) {
    int tid = tokens[l];
    if (tid < 0 || tid >= V) tid = mw::kPadId;
    for (int d = 0; d < D; ++d) {
      g_x[l * D + d] = mw::kTokEmb[tid * D + d] + mw::kPosEnc[l * D + d];
    }
  }

  run_encoder_layer(
      seq_len,
      mw::klayer0_norm1_w,
      mw::klayer0_norm1_b,
      mw::klayer0_in_proj_w,
      mw::klayer0_in_proj_b,
      mw::klayer0_out_proj_w,
      mw::klayer0_out_proj_b,
      mw::klayer0_norm2_w,
      mw::klayer0_norm2_b,
      mw::klayer0_linear1_w,
      mw::klayer0_linear1_b,
      mw::klayer0_linear2_w,
      mw::klayer0_linear2_b);

  run_encoder_layer(
      seq_len,
      mw::klayer1_norm1_w,
      mw::klayer1_norm1_b,
      mw::klayer1_in_proj_w,
      mw::klayer1_in_proj_b,
      mw::klayer1_out_proj_w,
      mw::klayer1_out_proj_b,
      mw::klayer1_norm2_w,
      mw::klayer1_norm2_b,
      mw::klayer1_linear1_w,
      mw::klayer1_linear1_b,
      mw::klayer1_linear2_w,
      mw::klayer1_linear2_b);

  float last[64];
  layer_norm(g_x + (seq_len - 1) * D, mw::kFinalNorm_w, mw::kFinalNorm_b, D, last);
  linear(last, D, mw::kHead_w, mw::kHead_b, V, logits_out);
}
