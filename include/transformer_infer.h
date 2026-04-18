#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Run full forward pass; writes logits for last token only (length V). */
void tr_forward_logits_last(const int* tokens, int seq_len, float* logits_out);

/** Map UTF-8 byte to vocab id; unknown -> space id. */
int tr_char_to_id(char c);

/** Decode ids to null-terminated string (caller provides buf, max_len includes NUL). */
void tr_decode_skip_special(const int* ids, int n_ids, char* buf, size_t buf_sz);

#ifdef __cplusplus
}
#endif
