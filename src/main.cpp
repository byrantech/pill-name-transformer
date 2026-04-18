/**
 * ESP32-S3-LCD-1.3 (Waveshare): ST7789 label UI + QMI8658 shake + on-device transformer inference.
 */
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <SensorQMI8658.hpp>
#include <TFT_eSPI.h>

#include "code128_patterns.h"
#include "model_weights_generated.h"
#include "transformer_infer.h"

namespace {

constexpr int kImuSda = 47;
constexpr int kImuScl = 48;
constexpr int kBlPin = 20;

constexpr float kGyroShakeDps = 150.f;
constexpr uint8_t kGyroHighStreak = 5;
constexpr uint32_t kShakeArmMs = 2000;
constexpr uint32_t kMinMsBetweenInfer = 2800;

#ifndef SHAKE_SERIAL_DEBUG
#define SHAKE_SERIAL_DEBUG 0
#endif

TFT_eSPI tft;
SensorQMI8658 qmi;
uint32_t g_shake_allowed_after_ms = 0;

/* ST7789 on Waveshare 1.3" often shows correct neutrals but shifts saturated hues (yellow→cyan,
 * red→blue). Swap RGB565's R and B fields so tft.color565(r,g,b) matches the panel. If colors
 * look wrong, set to 0 in platformio.ini: -DPANEL_SWAP_RB_IN_565=0 and try TFT_RGB_ORDER TFT_RGB
 * in include/waveshare_tft_eSPI_setup.h instead. */
#ifndef PANEL_SWAP_RB_IN_565
#define PANEL_SWAP_RB_IN_565 1
#endif
#if PANEL_SWAP_RB_IN_565
inline uint16_t panel565(uint16_t c) {
  return static_cast<uint16_t>((c & 0x07E0) | ((c & 0x001F) << 11) | ((c & 0xF800) >> 11));
}
#else
inline uint16_t panel565(uint16_t c) { return c; }
#endif
inline uint16_t panel_rgb(uint8_t r, uint8_t g, uint8_t b) { return panel565(tft.color565(r, g, b)); }

void backlight_on() {
  pinMode(kBlPin, OUTPUT);
  digitalWrite(kBlPin, HIGH);
}

inline float rand_open01() {
  const uint32_t r = esp_random();
  return (static_cast<float>(r) + 1.0f) * (1.0f / 4294967296.0f);
}

inline float random_gumbel() {
  const float u = rand_open01();
  return -logf(-logf(u));
}

int sample_full_vocab_gumbel(const float* logits, float temp) {
  const int V = mw::kVocabSize;
  int best = 0;
  float best_score = (logits[0] / temp) + random_gumbel();
  for (int i = 1; i < V; ++i) {
    const float s = (logits[i] / temp) + random_gumbel();
    if (s > best_score) {
      best_score = s;
      best = i;
    }
  }
  return best;
}

int sample_first_letter_gumbel(const float* logits, float temp) {
  int best = -1;
  float best_score = 0.f;
  for (int i = 0; i < mw::kVocabSize; ++i) {
    const char c = mw::kVocabChars[i];
    if (c < 'a' || c > 'z') continue;
    const float s = (logits[i] / temp) + random_gumbel();
    if (best < 0 || s > best_score) {
      best_score = s;
      best = i;
    }
  }
  if (best < 0) return sample_full_vocab_gumbel(logits, temp);
  return best;
}

void generate_name(char* out, size_t out_sz) {
  int seq[32];
  int L = 0;
  seq[L++] = mw::kBosId;
  while (L < mw::kSeqLen) {
    float logits[mw::kVocabSize];
    tr_forward_logits_last(seq, L, logits);
    const bool first_char = (L == 1);
    const float temp = first_char ? 1.15f : 0.92f;
    const int next =
        first_char ? sample_first_letter_gumbel(logits, temp) : sample_full_vocab_gumbel(logits, temp);
    seq[L++] = next;
    if (next == mw::kEosId) break;
  }
  tr_decode_skip_special(seq, L, out, out_sz);
}

bool init_imu() {
  Wire.begin(kImuSda, kImuScl);
  if (!qmi.begin(Wire, QMI8658_L_SLAVE_ADDRESS, kImuSda, kImuScl)) return false;
  qmi.configAccelerometer(SensorQMI8658::ACC_RANGE_4G, SensorQMI8658::ACC_ODR_125Hz,
                          SensorQMI8658::LPF_MODE_3);
  qmi.configGyroscope(SensorQMI8658::GYR_RANGE_256DPS, SensorQMI8658::GYR_ODR_112_1Hz,
                      SensorQMI8658::LPF_MODE_3);
  qmi.enableAccelerometer();
  qmi.enableGyroscope();
  return true;
}

// --- Label content -----------------------------------------------------------

const char* kMfg[] = {
    "PFIZER",        "MERCK",         "JANSSEN",       "GENENTECH",     "ROCHE",
    "NOVARTIS",      "ASTRAZENECA",   "BRISTOL MYERS", "ELI LILLY",     "ABBVIE",
    "GSK",           "SANOFI",        "BOEHRINGER",    "AMGEN",         "GILEAD",
    "BIOGEN",        "REGENERON",     "VERTEX",        "MODERNA",       "BAYER",
    "TAKEDA",        "ORGANON",       "VIATRIS",       "CIPLA",         "SUN PHARMA",
    "LUPIN",         "ZYDUS",         "APOTEX",        "ACCORD HLTH",   "FRESENIUS",
    "BAXTER",        "MALLINCKRODT",  "ALKERMES",      "SUMMIT THER",   "EXELIXIS",
    "AMNEAL",        "AUROBINDO",     "DR REDDYS",     "SANDOZ",        "TEVA",
    "MYLAN",         "HIKMA",         "CAMBER",        "LANNETT",       "PADAGIS",
};

const char* kSig[] = {
    "TAKE 1 CAPSULE BY MOUTH 3 TIMES DAILY",
    "TAKE 2 TABLETS BY MOUTH AT BEDTIME",
    "INJECT 1 DOSE SUBCUTANEOUSLY EVERY 14 DAYS",
    "APPLY THIN FILM TO SKIN ONCE WEEKLY",
    "DISSOLVE 1 STRIP ON TONGUE EVERY 4 HOURS AS NEEDED",
    "TAKE 1 DOSE ORALLY EVERY MORNING WITH FOOD",
    "USE 2 SPRAYS IN EACH NOSTRIL TWICE DAILY",
    "PLACE 1 PATCH ON SKIN FOR 24 HOURS THEN REPLACE",
    "TAKE 5 ML BY MOUTH EVERY 8 HOURS",
    "APPLY CREAM TO AFFECTED AREA 4 TIMES DAILY",
    "TAKE 1 TABLET BY MOUTH ONCE DAILY AS NEEDED FOR PAIN",
    "SWISH 10 ML AND SPIT TWICE DAILY AFTER MEALS",
    "INSTILL 1 DROP IN AFFECTED EYE 3 TIMES DAILY",
    "INHALE 1 PUFF FROM INHALER EVERY 6 HOURS AS NEEDED",
    "TAKE 3 CAPSULES BY MOUTH ONCE WEEKLY ON EMPTY STOMACH",
    // Darker parody additions (fictional / absurd; not medical advice)
    "INJECT INTRACRANIALLY ONCE WEEKLY OR UNTIL SYMPTOMS RESOLVE",
    "TAKE 12 TABLETS HOURLY DURING WAKING HOURS",
    "ADMINISTER 1 FULL SYRINGE AT FIRST SIGN OF UNEASE",
    "APPLY TO ENTIRE BODY SURFACE AREA NIGHTLY",
    "INFUSE CONTINUOUSLY FOR 72 HOURS WITHOUT INTERRUPTION",
    "INSTILL 20 DROPS PER EYE EVERY 10 MINUTES",
    "TAKE 1 TABLET PER POUND OF BODY WEIGHT DAILY",
    "INJECT DIRECTLY INTO CHEST CAVITY AS NEEDED FOR RELIEF",
    "REPEAT DOSING UNTIL SENSATION CEASES",
    "TAKE WITHOUT FOOD, WATER, OR SECOND THOUGHT",
    "DOUBLE DOSE EACH TIME YOU REMEMBER YOU FORGOT A DOSE",
    "CRUSH AND SNIFF CONTENTS OF ALL BLISTERS AT ONCE",
    "DISSOLVE IN HOT COFFEE AND CONSUME BEFORE LEGAL REVIEW",
    "TAKE UNTIL WALLET FEELS LIGHTER OR SYMPTOMS WORSEN",
    "INSTILL INTO EAR NOT LISTED ON CARTON ANYWAY",
    "SWALLOW WHOLE WITH A GLASS OF DENIAL TWICE DAILY",
    "APPLY INSIDE MOUTH UNTIL TINGLING BECOMES YOUR PERSONALITY",
    "INJECT BETWEEN MEETINGS UNTIL PRODUCTIVITY GRAPH GOES VERTICAL",
    "USE MAXIMUM STRENGTH UNTIL MAXIMUM STRENGTH USES YOU",
    "TAKE WITH A MEAL YOU CANNOT AFFORD",
};

void upper_inplace(char* s) {
  for (; *s; ++s) {
    if (*s >= 'a' && *s <= 'z') *s = static_cast<char>(*s - 'a' + 'A');
  }
}

void format_price_pow2(uint32_t exp, char* buf, size_t cap) {
  if (exp <= 12) {
    snprintf(buf, cap, "$%u", 1u << exp);
    return;
  }
  if (exp <= 62) {
    snprintf(buf, cap, "$%llu", static_cast<unsigned long long>(1ULL << exp));
    return;
  }
  const double log10p = static_cast<double>(exp) * 0.3010299956639812;
  const int e10 = static_cast<int>(floor(log10p));
  const double mant = pow(10.0, log10p - static_cast<double>(e10));
  snprintf(buf, cap, "~$%.1fe+%d", mant, e10);
}

void build_code128_payload(const char* rx, int qty, char* digits_out, size_t cap) {
  char acc[20];
  size_t j = 0;
  for (const char* p = rx; *p && j + 1 < sizeof(acc); ++p) {
    if (*p >= '0' && *p <= '9') acc[j++] = *p;
  }
  acc[j] = 0;
  char tail[6];
  snprintf(tail, sizeof(tail), "%02d", 1 + (qty % 99));
  snprintf(digits_out, cap, "%s%s", acc, tail);
  if (digits_out[0] == '\0') {
    strncpy(digits_out, "0123456788", cap - 1);
    digits_out[cap - 1] = '\0';
  }
  if (strlen(digits_out) > 14) digits_out[14] = '\0';
  if ((strlen(digits_out) & 1U) != 0) {
    const size_t L = strlen(digits_out);
    digits_out[L - 1] = '\0';
  }
  if ((strlen(digits_out) & 1U) != 0 && strlen(digits_out) + 2 < cap) {
    const size_t L = strlen(digits_out);
    memmove(digits_out + 1, digits_out, L + 1);
    digits_out[0] = '0';
  }
}

int code128_encode_c(const char* digits_even, uint8_t* out, int max_out) {
  const size_t len = strlen(digits_even);
  if (len < 2 || (len & 1U) != 0 || max_out < 4) return 0;
  int n = 0;
  out[n++] = 105;
  for (size_t i = 0; i + 1 < len && n < max_out - 2; i += 2) {
    if (digits_even[i] < '0' || digits_even[i] > '9') return 0;
    if (digits_even[i + 1] < '0' || digits_even[i + 1] > '9') return 0;
    out[n++] = static_cast<uint8_t>((digits_even[i] - '0') * 10 + (digits_even[i + 1] - '0'));
  }
  int sum = out[0];
  for (int i = 1; i < n; ++i) sum += out[i] * i;
  out[n++] = static_cast<uint8_t>(sum % 103);
  return n;
}

void draw_code128_strip(int16_t x, int16_t y, int16_t max_w, int16_t bar_h, const char* digits_human,
                        uint16_t paper_bg) {
  uint8_t sym[24];
  const int ns = code128_encode_c(digits_human, sym, 24);
  if (ns <= 0) return;

  int total_mod = 20;
  for (int i = 0; i < ns; ++i) {
    for (int j = 0; j < 6; ++j) total_mod += kCode128Pat[sym[i]][j];
  }
  for (int j = 0; j < 7; ++j) total_mod += kCode128Stop[j];

  const int modw = (total_mod > 0) ? (max_w / total_mod) : 1;
  const int mod = (modw < 1) ? 1 : modw;

  tft.fillRect(x, y, max_w, static_cast<int16_t>(bar_h + 14), paper_bg);
  tft.drawFastHLine(x, y, max_w, panel_rgb(160, 160, 162));

  int cx = x + 10 * mod;
  bool bar_on = true;
  for (int i = 0; i < ns; ++i) {
    for (int j = 0; j < 6; ++j) {
      const int w = kCode128Pat[sym[i]][j] * mod;
      if (bar_on) tft.fillRect(cx, static_cast<int16_t>(y + 2), w, bar_h, TFT_BLACK);
      cx += w;
      bar_on = !bar_on;
    }
  }
  for (int j = 0; j < 7; ++j) {
    const int w = kCode128Stop[j] * mod;
    if (bar_on) tft.fillRect(cx, static_cast<int16_t>(y + 2), w, bar_h, TFT_BLACK);
    cx += w;
    bar_on = !bar_on;
  }

  tft.setTextFont(1);
  tft.setTextSize(1);
  tft.setTextDatum(TC_DATUM);
  tft.setTextColor(panel_rgb(28, 28, 30), paper_bg);
  tft.drawString(digits_human, x + max_w / 2, static_cast<int16_t>(y + bar_h + 6));
  tft.setTextDatum(TL_DATUM);
}

void draw_wrapped_sig(const char* text, int16_t x, int16_t y, int16_t x_max, int16_t y_max, uint8_t text_size,
                      uint16_t bg) {
  tft.setTextFont(1);
  tft.setTextSize(text_size);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(panel_rgb(28, 28, 30), bg);
  char buf[96];
  strncpy(buf, text, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  const int line_h = 8 * static_cast<int>(text_size) + 2;
  const int max_w_px = x_max - x;
  int cy = y;
  char line_acc[96];
  line_acc[0] = '\0';
  char* saveptr = nullptr;
  char* tok = strtok_r(buf, " ", &saveptr);

  while (tok && cy + line_h <= y_max) {
    char trial[96];
    if (line_acc[0] != '\0') {
      snprintf(trial, sizeof(trial), "%s %s", line_acc, tok);
    } else {
      snprintf(trial, sizeof(trial), "%s", tok);
    }
    if (tft.textWidth(trial) <= max_w_px) {
      strncpy(line_acc, trial, sizeof(line_acc) - 1);
      line_acc[sizeof(line_acc) - 1] = '\0';
    } else {
      if (line_acc[0] != '\0') {
        tft.drawString(line_acc, x, cy);
        cy += line_h;
        strncpy(line_acc, tok, sizeof(line_acc) - 1);
        line_acc[sizeof(line_acc) - 1] = '\0';
      } else {
        tft.drawString(tok, x, cy);
        cy += line_h;
        line_acc[0] = '\0';
      }
    }
    tok = strtok_r(nullptr, " ", &saveptr);
  }
  if (line_acc[0] != '\0' && cy + line_h <= y_max) {
    tft.drawString(line_acc, x, cy);
  }
}

void draw_drug_name_fit(const char* name, int16_t x, int16_t y, int16_t max_w, uint16_t bg) {
  tft.setTextDatum(TL_DATUM);
  tft.setTextFont(1);
  char line[96];
  strncpy(line, name, sizeof(line) - 1);
  line[sizeof(line) - 1] = 0;
  upper_inplace(line);
  uint8_t sz = 3;
  tft.setTextSize(sz);
  int tw = tft.textWidth(line);
  if (tw > max_w) {
    sz = 2;
    tft.setTextSize(sz);
    tw = tft.textWidth(line);
  }
  if (tw > max_w) {
    sz = 1;
    tft.setTextSize(sz);
    tw = tft.textWidth(line);
    if (tw > max_w) {
      while (strlen(line) > 4 && tft.textWidth(line) > max_w) line[strlen(line) - 1] = 0;
      strcat(line, "...");
    }
  }
  tft.setTextColor(panel_rgb(20, 20, 22), bg);
  tft.drawString(line, x, y);
}

void draw_label(const char* patient_upper, const char* date_mmddyyyy, const char* drug_name,
                const char* mfg, const char* sig, const char* rx_line, const char* exp_line, int qty,
                const char* price_str, const char* barcode_digits) {
  /* Walgreens-adjacent retail Rx label palette (thermal paper, highlighter yellow, brand red). */
  const uint16_t paper = panel_rgb(252, 250, 247);
  const uint16_t wag_red = panel_rgb(227, 24, 55);  // #E31837
  const uint16_t hilite = panel_rgb(255, 236, 52);
  const uint16_t ink = panel_rgb(24, 24, 26);
  const uint16_t muted = panel_rgb(95, 95, 98);

  tft.fillScreen(paper);
  tft.setTextDatum(TL_DATUM);
  tft.setTextFont(1);

  tft.setTextColor(ink, paper);
  tft.setTextSize(2);
  tft.drawString(patient_upper, 4, 2);
  tft.setTextSize(1);
  tft.setTextDatum(TR_DATUM);
  tft.setTextColor(muted, paper);
  tft.drawString(date_mmddyyyy, tft.width() - 4, 6);
  tft.setTextDatum(TL_DATUM);

  constexpr int rx_left = 4;
  constexpr int rx_top = 22;
  constexpr int rx_w = 232;
  constexpr int rx_h = 98;
  tft.drawRect(rx_left, rx_top, rx_w, rx_h, panel_rgb(55, 55, 58));

  draw_drug_name_fit(drug_name, 8, 28, 216, paper);
  char mfg_line[48];
  snprintf(mfg_line, sizeof(mfg_line), "MFG %s", mfg);
  tft.setTextSize(1);
  tft.setTextColor(ink, paper);
  tft.drawString(mfg_line, 8, 56);
  draw_wrapped_sig(sig, 8, 68, 228, 116, 1, paper);

  constexpr int y_rx_row = 124;
  constexpr int h_row = 20;
  tft.fillRect(4, y_rx_row, 232, h_row, hilite);
  tft.setTextColor(ink, hilite);
  tft.setTextSize(1);
  tft.drawString(rx_line, 6, y_rx_row + 6);
  tft.setTextDatum(TR_DATUM);
  tft.drawString(exp_line, tft.width() - 6, y_rx_row + 6);
  tft.setTextDatum(TL_DATUM);

  constexpr int y_qty_row = y_rx_row + h_row;
  constexpr int h_qty_row = 30;
  tft.fillRect(4, y_qty_row, 232, h_qty_row, hilite);
  tft.setTextColor(ink, hilite);
  char qty_line[24];
  snprintf(qty_line, sizeof(qty_line), "QTY %d", qty);
  tft.setTextSize(1);
  tft.drawString(qty_line, 6, y_qty_row + 11);
  char price_line[56];
  snprintf(price_line, sizeof(price_line), "YOUR COST %s", price_str);
  tft.setTextSize(2);
  tft.setTextDatum(TR_DATUM);
  tft.drawString(price_line, tft.width() - 6, y_qty_row + 9);
  tft.setTextDatum(TL_DATUM);
  tft.setTextSize(1);

  constexpr int y_bc = 175;
  draw_code128_strip(4, y_bc, 232, 14, barcode_digits, paper);

  constexpr int y_logo = 205;
  constexpr int h_logo = 25;
  tft.fillRect(0, y_logo, tft.width(), h_logo, paper);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(wag_red, paper);
  tft.setTextSize(2);
  tft.drawString("WALGREENS", 8, static_cast<int16_t>(y_logo + 4));

  tft.setTextDatum(TL_DATUM);
  tft.setTextSize(1);
  tft.setTextColor(muted, paper);
  tft.drawString("Thank you", 4, 232);
  tft.setTextDatum(TR_DATUM);
  tft.drawString("Rx ONLY", tft.width() - 4, 232);
  tft.setTextDatum(TL_DATUM);
}

void random_patient(char* out, size_t cap) {
  const int n = 6 + static_cast<int>(esp_random() % 5);
  size_t i = 0;
  for (; i < static_cast<size_t>(n) && i + 1 < cap; ++i) {
    const char* set = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    out[i] = set[esp_random() % (strlen(set))];
  }
  out[i] = 0;
}

void random_rx(char* out, size_t cap) {
  const unsigned a = esp_random() % 9000000u + 1000000u;
  const unsigned b = esp_random() % 90000u + 10000u;
  snprintf(out, cap, "RX %u-%u", a, b);
}

void boot_label_strings(char* patient, char* date, char* rx) {
  random_patient(patient, 16);
  upper_inplace(patient);
  const unsigned mo = esp_random() % 12 + 1;
  const unsigned dy = esp_random() % 28 + 1;
  const unsigned yr = 2077;
  snprintf(date, 16, "%02u/%02u/%04u", mo, dy, yr);
  random_rx(rx, 32);
}

}  // namespace

static char g_patient[16];
static char g_date[16];
static char g_rx[36];
static char g_drug[96];
static char g_sig[96];
static char g_price[32];
static char g_barcode_digits[20];
static int g_qty = 1;

void setup() {
  Serial.begin(115200);
  delay(200);

  backlight_on();
  tft.init();
  tft.setRotation(0);

  const uint16_t boot_paper = panel_rgb(252, 250, 247);
  const uint16_t boot_red = panel_rgb(227, 24, 55);
  tft.fillScreen(boot_paper);
  tft.setTextDatum(BL_DATUM);
  tft.setTextColor(boot_red, boot_paper);
  tft.setTextSize(2);
  tft.drawString("WALGREENS", 6, tft.height() - 8);
  tft.setTextDatum(MC_DATUM);
  tft.setTextSize(1);
  tft.setTextColor(panel_rgb(95, 95, 98), boot_paper);
  tft.drawString("Loading Rx display...", tft.width() / 2, tft.height() / 2 + 8);

  if (!init_imu()) {
    tft.fillScreen(boot_paper);
    tft.setTextColor(boot_red, boot_paper);
    tft.setTextSize(2);
    tft.drawString("SENSOR ERROR", tft.width() / 2, tft.height() / 2 - 8);
    tft.setTextSize(1);
    tft.setTextColor(panel_rgb(55, 55, 58), boot_paper);
    tft.drawString("QMI8658 not found", tft.width() / 2, tft.height() / 2 + 14);
    while (true) delay(1000);
  }

  boot_label_strings(g_patient, g_date, g_rx);
  strncpy(g_sig, kSig[esp_random() % (sizeof(kSig) / sizeof(kSig[0]))], sizeof(g_sig) - 1);
  g_sig[sizeof(g_sig) - 1] = 0;

  generate_name(g_drug, sizeof(g_drug));
  format_price_pow2(0, g_price, sizeof(g_price));
  g_qty = 1 + static_cast<int>(esp_random() % 100);
  build_code128_payload(g_rx, g_qty, g_barcode_digits, sizeof(g_barcode_digits));

  const char* mfg = kMfg[esp_random() % (sizeof(kMfg) / sizeof(kMfg[0]))];
  draw_label(g_patient, g_date, g_drug, mfg, g_sig, g_rx, "NEVER", g_qty, g_price, g_barcode_digits);

  g_shake_allowed_after_ms = millis() + kShakeArmMs;
  Serial.println("Ready — hold still ~2s, then shake to generate (cooldown ~2.8s).");
  Serial.print("drug: ");
  Serial.println(g_drug);
}

void loop() {
  static uint32_t last_shake_ms = 0;
  static uint32_t next_infer_ok_ms = 0;
  static uint8_t gyro_high_streak = 0;
  static bool infer_busy = false;
  static uint32_t price_exp = 0;

#if SHAKE_SERIAL_DEBUG
  static uint32_t last_dbg_ms = 0;
#endif

  IMUdata gyr;
  if (!qmi.getGyroscope(gyr.x, gyr.y, gyr.z)) {
    delay(5);
    return;
  }

  const float spin = sqrtf(gyr.x * gyr.x + gyr.y * gyr.y + gyr.z * gyr.z);
  const uint32_t now = millis();

#if SHAKE_SERIAL_DEBUG
  if (now - last_dbg_ms > 500) {
    last_dbg_ms = now;
    Serial.printf("imu spin=%.1f dps\n", static_cast<double>(spin));
  }
#endif

  if (static_cast<int32_t>(now - g_shake_allowed_after_ms) < 0 || infer_busy ||
      now < next_infer_ok_ms) {
    if (spin <= kGyroShakeDps) gyro_high_streak = 0;
    delay(15);
    return;
  }

  if (spin > kGyroShakeDps) {
    ++gyro_high_streak;
  } else {
    gyro_high_streak = 0;
  }

  if (gyro_high_streak < kGyroHighStreak) {
    delay(15);
    return;
  }
  if ((now - last_shake_ms) <= 850) {
    delay(15);
    return;
  }

  gyro_high_streak = 0;
  last_shake_ms = now;
  infer_busy = true;

  tft.fillRect(8, 88, 200, 16, panel_rgb(255, 255, 255));
  tft.setTextDatum(TL_DATUM);
  tft.setTextFont(1);
  tft.setTextSize(2);
  tft.setTextColor(panel_rgb(255, 0, 0), panel_rgb(255, 255, 255));
  tft.drawString("GENERATING...", 10, 90);

  generate_name(g_drug, sizeof(g_drug));
  ++price_exp;
  format_price_pow2(price_exp, g_price, sizeof(g_price));
  strncpy(g_sig, kSig[esp_random() % (sizeof(kSig) / sizeof(kSig[0]))], sizeof(g_sig) - 1);
  g_sig[sizeof(g_sig) - 1] = 0;
  random_rx(g_rx, sizeof(g_rx));
  g_qty = 1 + static_cast<int>(esp_random() % 100);
  build_code128_payload(g_rx, g_qty, g_barcode_digits, sizeof(g_barcode_digits));

  const char* mfg = kMfg[esp_random() % (sizeof(kMfg) / sizeof(kMfg[0]))];
  draw_label(g_patient, g_date, g_drug, mfg, g_sig, g_rx, "NEVER", g_qty, g_price, g_barcode_digits);

  infer_busy = false;
  next_infer_ok_ms = millis() + kMinMsBetweenInfer;

  Serial.print("shake -> ");
  Serial.print(g_drug);
  Serial.print(" | ");
  Serial.println(g_price);

  delay(15);
}
