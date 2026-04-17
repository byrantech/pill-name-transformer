#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "model_weights_generated.h"
#include "transformer_infer.h"

namespace {

int sample_topk(const float* logits, float temp, int topk) {
  const int V = mw::kVocabSize;
  int order[32];
  for (int i = 0; i < V; ++i) order[i] = i;
  for (int a = 0; a < V; ++a) {
    for (int b = a + 1; b < V; ++b) {
      if (logits[order[b]] > logits[order[a]]) {
        int t = order[a];
        order[a] = order[b];
        order[b] = t;
      }
    }
  }
  int k = topk < V ? topk : V;
  if (k < 1) k = 1;
  float scaled[32];
  float maxlog = -1e30f;
  for (int i = 0; i < k; ++i) {
    scaled[i] = logits[order[i]] / temp;
    if (scaled[i] > maxlog) maxlog = scaled[i];
  }
  float sum = 0.f;
  for (int i = 0; i < k; ++i) {
    scaled[i] = expf(scaled[i] - maxlog);
    sum += scaled[i];
  }
  const float inv = sum > 0.f ? 1.f / sum : 1.f;
  const float r = static_cast<float>(esp_random()) / 4294967296.0f;
  float acc = 0.f;
  for (int i = 0; i < k; ++i) {
    acc += scaled[i] * inv;
    if (r <= acc) return order[i];
  }
  return order[k - 1];
}

void generate_name(const String& prompt, char* out, size_t out_sz) {
  int seq[32];
  int L = 0;
  seq[L++] = mw::kBosId;
  for (unsigned i = 0; i < prompt.length() && L < mw::kSeqLen - 1; ++i) {
    seq[L++] = tr_char_to_id(static_cast<char>(prompt[i]));
  }
  while (L < mw::kSeqLen) {
    float logits[mw::kVocabSize];
    tr_forward_logits_last(seq, L, logits);
    const int next = sample_topk(logits, 0.9f, 8);
    seq[L++] = next;
    if (next == mw::kEosId) break;
  }
  tr_decode_skip_special(seq, L, out, out_sz);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("pill-name-transformer (on-device weights) — type a line + Enter to generate.");
  Serial.println("Optional prefix after first char: p myprefix");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  String prompt;
  if (line.length() >= 2 && line[0] == 'p' && line[1] == ' ') {
    prompt = line.substring(2);
    prompt.trim();
  }

  char out[96];
  generate_name(prompt, out, sizeof(out));
  Serial.print("generated: ");
  Serial.println(out);
}
