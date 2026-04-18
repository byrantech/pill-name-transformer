/**
 * TFT_eSPI settings for Waveshare ESP32-S3-LCD-1.3 (ST7789V2, 240x240).
 * Pinout from Waveshare schematic (ESP32S3_1.3inch): MOSI=41, SCLK=40, CS=39,
 * DC=38, RST=42, BL_PWM=20. IMU QMI8658 uses I2C SDA=47, SCL=48 (not used here).
 *
 * Included via PlatformIO: -DUSER_SETUP_LOADED -include include/waveshare_tft_eSPI_setup.h
 */
#pragma once

// ESP32-S3: Arduino core defines FSPI as 0, but SoC REG_SPI_BASE(i) is only valid for GPSPI
// host indices 2 and 3 (see soc/spi_reg.h). SPI_PORT 0 yields REG_SPI_BASE=0 and a crash in
// TFT_eSPI (StoreProhibited at 0x10 = SPI_USER_REG). USE_HSPI_PORT selects SPI_PORT 3 on S3.
#define USE_HSPI_PORT

#define ST7789_DRIVER
#define TFT_WIDTH 240
#define TFT_HEIGHT 240

/* MADCTL colour order for this ST7789. If hues are still wrong after flashing, toggle this line
 * between TFT_RGB and TFT_BGR and/or set -DPANEL_SWAP_RB_IN_565=0 in platformio.ini build_flags. */
#define TFT_RGB_ORDER TFT_BGR
#define TFT_INVERSION_ON

#define TFT_MISO -1
#define TFT_MOSI 41
#define TFT_SCLK 40
#define TFT_CS 39
#define TFT_DC 38
#define TFT_RST 42

#define TFT_BL 20
#define TFT_BACKLIGHT_ON HIGH

#define TOUCH_CS -1

#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_FONT4
#define LOAD_FONT6
#define LOAD_FONT7
#define LOAD_FONT8
#define LOAD_GFXFF
#define SMOOTH_FONT

#define SPI_FREQUENCY 40000000
#define SPI_READ_FREQUENCY 16000000
