/*
 * ============================================================
 * CAMERA NODE v2 — No Hardware Modification Required
 * ============================================================
 * WIRING (SX1276/SX1278 → ESP32-CAM):
 *   LoRa VCC    → 3.3V (from Gateway ESP32 recommended)
 *   LoRa GND    → GND  (common with ESP32-CAM GND)
 *   LoRa MOSI   → GPIO 2
 *   LoRa SCK    → GPIO 14
 *   LoRa MISO   → GPIO 13
 *   LoRa NSS    → GPIO 15
 *   LoRa NRESET → NOT CONNECTED
 *   LoRa DIO0   → NOT CONNECTED
 *
 * NO hardware modification needed.
 * White LED on GPIO4 remains working.
 *
 * NOTE: GPIO14 is the SD_MMC CLK pin. Since we are not
 * using SD card this is completely free and safe to use.
 *
 * ARDUINO IDE SETTINGS:
 *   Board            : AI Thinker ESP32-CAM
 *   Partition Scheme : Huge APP (3MB No OTA)
 *   PSRAM            : Enabled
 *   Upload Speed     : 115200
 * ============================================================
 */

#include <Arduino.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include <SPI.h>
#include <LoRa.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "acs_camera_model.h"

// ============================================================
// PIN DEFINITIONS — VERSION 2 (No Hardware Modification)
// ============================================================
#define LORA_SCK    14    // GPIO14 — SD_MMC CLK, free without SD
#define LORA_MISO   13    // GPIO13 — SD_MMC DATA3, free without SD
#define LORA_MOSI    2    // GPIO2
#define LORA_NSS    15    // GPIO15 — SD_MMC CMD, free without SD
#define LORA_RESET  -1    // not connected
#define LORA_DIO0   -1    // not connected

#define LORA_FREQ   433E6  // change to 868E6 or 915E6 if needed

#define FLASH_LED_PIN  4   // still available in this version

// Camera pins — AI Thinker (do not change)
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

// ============================================================
// MODEL PARAMETERS
// ============================================================
#define IMG_WIDTH        96
#define IMG_HEIGHT       96
#define NUM_CLASSES       3
#define CONFIDENCE_THRESH 0.70f

const char* CLASS_LABELS[NUM_CLASSES] = {
    "ANIMAL",
    "BACKGROUND",
    "HUMAN"
};

// ============================================================
// TFLITE GLOBALS
// ============================================================
const int   ARENA_SIZE   = 512 * 1024;
uint8_t*    tensor_arena = nullptr;

const tflite::Model*      tfl_model     = nullptr;
tflite::MicroInterpreter* interpreter   = nullptr;
TfLiteTensor*             input_tensor  = nullptr;
TfLiteTensor*             output_tensor = nullptr;

struct InferenceResult {
    int   class_id;
    float confidence;
    char  label[16];
    bool  alert;
};

// ============================================================
// CAMERA INIT
// ============================================================
bool init_camera() {
    camera_config_t cfg;
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0       = CAM_PIN_D0;
    cfg.pin_d1       = CAM_PIN_D1;
    cfg.pin_d2       = CAM_PIN_D2;
    cfg.pin_d3       = CAM_PIN_D3;
    cfg.pin_d4       = CAM_PIN_D4;
    cfg.pin_d5       = CAM_PIN_D5;
    cfg.pin_d6       = CAM_PIN_D6;
    cfg.pin_d7       = CAM_PIN_D7;
    cfg.pin_xclk     = CAM_PIN_XCLK;
    cfg.pin_pclk     = CAM_PIN_PCLK;
    cfg.pin_vsync    = CAM_PIN_VSYNC;
    cfg.pin_href     = CAM_PIN_HREF;
    cfg.pin_sscb_sda = CAM_PIN_SIOD;
    cfg.pin_sscb_scl = CAM_PIN_SIOC;
    cfg.pin_pwdn     = CAM_PIN_PWDN;
    cfg.pin_reset    = CAM_PIN_RESET;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_RGB565;
    cfg.frame_size   = FRAMESIZE_96X96;
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Failed: 0x%x\n", err);
        return false;
    }
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    Serial.println("[CAM] OK");
    return true;
}

// ============================================================
// TFLITE INIT
// ============================================================
bool init_tflite() {
    tensor_arena = (uint8_t*)heap_caps_malloc(
        ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!tensor_arena) {
        Serial.println("[TFL] Arena alloc failed");
        return false;
    }

    tfl_model = tflite::GetModel(acs_camera_model);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFL] Schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddAdd();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddGreater();
    resolver.AddCast();
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interp(
        tfl_model, resolver, tensor_arena, ARENA_SIZE
    );
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFL] AllocateTensors failed");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.printf("[TFL] Input: [%d,%d,%d,%d] OK\n",
        input_tensor->dims->data[0],
        input_tensor->dims->data[1],
        input_tensor->dims->data[2],
        input_tensor->dims->data[3]);
    return true;
}

// ============================================================
// PREPROCESS
// ============================================================
void preprocess(camera_fb_t* fb, int8_t* buf) {
    int idx = 0;
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            int      pi  = (y * IMG_WIDTH + x) * 2;
            uint16_t px  = (fb->buf[pi] << 8) | fb->buf[pi + 1];
            uint8_t  r   = ((px >> 11) & 0x1F) << 3;
            uint8_t  g   = ((px >>  5) & 0x3F) << 2;
            uint8_t  b   = ( px        & 0x1F) << 3;
            buf[idx++]   = (int8_t)(r - 128);
            buf[idx++]   = (int8_t)(g - 128);
            buf[idx++]   = (int8_t)(b - 128);
        }
    }
}

// ============================================================
// INFERENCE
// ============================================================
InferenceResult run_inference() {
    InferenceResult res = {-1, 0.0f, "UNKNOWN", false};

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[INF] Capture failed");
        return res;
    }

    preprocess(fb, input_tensor->data.int8);
    esp_camera_fb_return(fb);

    unsigned long t0 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[INF] Invoke failed");
        return res;
    }
    unsigned long t_ms = millis() - t0;

    float probs[NUM_CLASSES];
    float total      = 0;
    float scale      = output_tensor->params.scale;
    int   zero_point = output_tensor->params.zero_point;

    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = (output_tensor->data.int8[i] - zero_point) * scale;
        if (probs[i] < 0) probs[i] = 0;
        total += probs[i];
    }
    if (total > 0)
        for (int i = 0; i < NUM_CLASSES; i++) probs[i] /= total;

    int   best_idx  = 0;
    float best_conf = probs[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > best_conf) {
            best_conf = probs[i];
            best_idx  = i;
        }
    }

    res.class_id   = best_idx;
    res.confidence = best_conf;
    strncpy(res.label, CLASS_LABELS[best_idx], 15);
    res.alert = (best_idx != 1) && (best_conf >= CONFIDENCE_THRESH);

    Serial.printf("[INF] %s %.1f%% | %lums\n",
                  res.label, best_conf * 100, t_ms);
    return res;
}

// ============================================================
// SEND LORA
// ============================================================
void send_lora(const InferenceResult& res) {
    char payload[32];
    snprintf(payload, sizeof(payload),
             "CAM|N01|%s|%.2f", res.label, res.confidence);

    LoRa.beginPacket();
    LoRa.print(payload);
    int ok = LoRa.endPacket();

    Serial.printf("[LORA TX] %s | %s\n",
                  payload, ok ? "OK" : "FAILED");

    // Flash LED to confirm transmission
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(100);
    digitalWrite(FLASH_LED_PIN, LOW);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[SYS] Camera Node v2 (No Hardware Mod) Booting...");

    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);

    // ── LoRa FIRST ─────────────────────────────────────────
    // GPIO14=SCK, GPIO13=MISO, GPIO2=MOSI, GPIO15=NSS
    // No transistor removal needed
    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
    LoRa.setPins(LORA_NSS, LORA_RESET, LORA_DIO0);

    if (!LoRa.begin(LORA_FREQ)) {
        Serial.println("[LORA] FAILED — check wiring:");
        Serial.println("       MOSI→GPIO2   SCK→GPIO14");
        Serial.println("       MISO→GPIO13  NSS→GPIO15");
        Serial.println("       RST→NC       VCC→3.3V");
        while (1) delay(1000);
    }
    LoRa.setSyncWord(0xF3);    // must match gateway
    LoRa.setTxPower(2);        // low power — reduces brownout

    Serial.println("[LORA] OK");

    // ── Camera ─────────────────────────────────────────────
    if (!init_camera()) {
        Serial.println("[CAM] FAILED");
        while (1) delay(1000);
    }

    // ── TFLite ─────────────────────────────────────────────
    if (!init_tflite()) {
        Serial.println("[TFL] FAILED");
        while (1) delay(1000);
    }

    Serial.printf("[SYS] Free PSRAM: %d bytes\n", ESP.getFreePsram());
    Serial.println("[SYS] Ready — inference every 5 seconds\n");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
    Serial.println("[SYS] Capturing...");

    InferenceResult res = run_inference();

    if (res.alert) {
        send_lora(res);
    } else {
        Serial.printf("[SYS] No alert: %s %.1f%%\n",
                      res.label, res.confidence * 100);
    }

    delay(5000);
}
