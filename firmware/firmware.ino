/**
 * @file firmware.ino
 * @brief Standalone Dual-Core Assistive Eyewear Firmware for ESP32-CAM using FOMO.
 * 
 * Perception Architecture:
 * - Core 0: 30-50 Hz High-Speed 3x VL53L1X ToF Ranging Reflex (< 30ms reaction)
 * - Core 1: Pure FOMO (Faster Objects, More Objects) MobileNetV2 Centroid Detection
 *           Running at 10-15 FPS on ESP32-CAM.
 * 
 * Target: ESP32-CAM (AI-Thinker) Standalone Eyewear
 */

#include <Arduino.h>
#include "esp_camera.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Project Components
#include "model_data.h"
#include "tof_array.h"
#include "feedback_manager.h"

// --- Camera Pin Definitions (AI-Thinker ESP32-CAM) ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// --- FOMO Model Parameters ---
#define FOMO_INPUT_W       96
#define FOMO_INPUT_H       96
#define FOMO_GRID_W        12
#define FOMO_GRID_H        12
#define FOMO_NUM_CLASSES   11

// Class Indices (Index 0 is Background)
enum FOMOClassID {
    FOMO_BG = 0,
    FOMO_ROAD = 1,
    FOMO_SIDEWALK = 2,
    FOMO_PERSON = 3,
    FOMO_VEHICLE = 4,
    FOMO_BOUNDARY = 5,
    FOMO_OBSTACLE = 6,
    FOMO_VEGETATION = 7,
    FOMO_STAIRS = 8,
    FOMO_CROSSWALK = 9,
    FOMO_UNEVEN_PAVEMENT = 10
};

// Global Subsystems
ToFArray tofArray;
FeedbackManager feedback;

// Shared Inter-Core State
volatile ToFReading g_lastTofReading;
portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;

// TFLite Micro Globals
constexpr int kTensorArenaSize = 120 * 1024; // 120 KB fits in internal SRAM
uint8_t* tensor_arena = nullptr;

const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// FreeRTOS Task Handles
TaskHandle_t TaskToFHandle = NULL;
TaskHandle_t TaskVisionHandle = NULL;


// ============================================================================
// CAMERA INITIALIZATION (NATIVE 96x96 FOR FOMO)
// ============================================================================
bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565; // RGB format
    config.frame_size = FRAMESIZE_96X96;     // Exact native hardware support on OV2640!
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[Camera] Init failed with error 0x%x\n", err);
        return false;
    }
    Serial.println("[Camera] OV2640 native 96x96 capture ready for FOMO.");
    return true;
}


// ============================================================================
// TFLITE MICRO INITIALIZATION
// ============================================================================
bool initTFLM() {
    Serial.println("[FOMO] Initializing TFLite Micro Engine...");

    tflite_model = tflite::GetModel(g_model);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[FOMO] Model schema mismatch!");
        return false;
    }

    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    if (!tensor_arena && psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    }
    if (!tensor_arena) {
        Serial.println("[FOMO] Tensor arena allocation failed!");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[FOMO] AllocateTensors() failed!");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("[FOMO] Model ready! Input: %dx%dx%d, Output: %dx%dx%d\n",
        input_tensor->dims->data[1], input_tensor->dims->data[2], input_tensor->dims->data[3],
        output_tensor->dims->data[1], output_tensor->dims->data[2], output_tensor->dims->data[3]);
    return true;
}


// ============================================================================
// CORE 0 TASK: HIGH-SPEED TOF REFLEX LOOP (30-50 Hz)
// ============================================================================
void TaskToFReflex(void* parameter) {
    Serial.println("[TaskToF] Core 0 Reflex Loop Active...");
    for (;;) {
        ToFReading r = tofArray.update();

        taskENTER_CRITICAL(&g_mux);
        g_lastTofReading = r;
        taskEXIT_CRITICAL(&g_mux);

        // Immediate Safety Reflex (< 30ms latency)
        if (r.overheadHazard) {
            // White canes cannot detect hanging branches or signs: Emergency chirp!
            feedback.triggerAlert(ALERT_OVERHEAD_CRITICAL, r.distTopMm);
        } else if (r.forwardObstacle && r.distMidMm < 1200) {
            feedback.triggerAlert(ALERT_FORWARD_OBSTACLE, r.distMidMm);
        }

        vTaskDelay(pdMS_TO_TICKS(25)); // 40 Hz loop rate
    }
}


// ============================================================================
// CORE 1 TASK: FOMO CENTROID INFERENCE LOOP (10-15 FPS)
// ============================================================================
void TaskVisionInference(void* parameter) {
    Serial.println("[TaskFOMO] Core 1 Centroid Perception Active...");
    int8_t* in_data = input_tensor->data.int8;

    for (;;) {
        uint32_t t_start = millis();

        // 1. Capture Camera Frame
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        // 2. Preprocess 96x96 RGB565 to INT8 [-128, 127]
        uint16_t* rgb565 = (uint16_t*)fb->buf;
        int num_pixels = FOMO_INPUT_W * FOMO_INPUT_H;
        int idx = 0;
        for (int p = 0; p < num_pixels; p++) {
            uint16_t pixel = rgb565[p];
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
            uint8_t b = (pixel & 0x1F) << 3;

            in_data[idx++] = (int8_t)((int)r - 128);
            in_data[idx++] = (int8_t)((int)g - 128);
            in_data[idx++] = (int8_t)((int)b - 128);
        }
        esp_camera_fb_return(fb);

        // 3. Run FOMO Inference
        if (interpreter->Invoke() == kTfLiteOk) {
            uint32_t t_infer = millis() - t_start;
            int8_t* out_data = output_tensor->data.int8;

            // 4. Parse 12x12 Centroid Grid
            int roadCentroids = 0, sidewalkCentroids = 0, rightBoundaryCentroids = 0;
            bool overheadVegetation = false;
            bool stairsDetected = false, crosswalkDetected = false, unevenDetected = false;

            for (int gy = 0; gy < FOMO_GRID_H; gy++) {
                for (int gx = 0; gx < FOMO_GRID_W; gx++) {
                    int cell_offset = (gy * FOMO_GRID_W + gx) * FOMO_NUM_CLASSES;

                    // Argmax across classes for this centroid cell
                    int topClass = 0;
                    int8_t maxScore = -128;
                    for (int c = 0; c < FOMO_NUM_CLASSES; c++) {
                        if (out_data[cell_offset + c] > maxScore) {
                            maxScore = out_data[cell_offset + c];
                            topClass = c;
                        }
                    }

                    // Ignore background (class 0)
                    if (topClass == FOMO_BG) continue;

                    // A. Overhead Hazard: Upper field of view (gy <= 3)
                    if (gy <= 3 && (topClass == FOMO_VEGETATION || topClass == FOMO_OBSTACLE)) {
                        overheadVegetation = true;
                    }

                    // B. Ground Walking Zone & Shorelining (gy >= 6)
                    if (gy >= 6) {
                        if (topClass == FOMO_ROAD) roadCentroids++;
                        if (topClass == FOMO_SIDEWALK) sidewalkCentroids++;
                        if (gx >= 9 && topClass == FOMO_BOUNDARY) rightBoundaryCentroids++;
                        if (topClass == FOMO_STAIRS) stairsDetected = true;
                        if (topClass == FOMO_CROSSWALK) crosswalkDetected = true;
                        if (topClass == FOMO_UNEVEN_PAVEMENT) unevenDetected = true;
                    }
                }
            }

            // 5. Sensor-Fused Feedback Decisions
            if (overheadVegetation) {
                feedback.triggerAlert(ALERT_OVERHEAD_CRITICAL, g_lastTofReading.distTopMm);
            } else if (roadCentroids >= 6) {
                // Drifting into vehicular road: Vibrate Left motor (steer right!)
                feedback.triggerAlert(ALERT_ROAD_DRIFT);
            } else if (rightBoundaryCentroids >= 2) {
                // Shoreline confidence pulse: wall/fence detected on right
                feedback.pulseShorelineConfidence();
            }

            // Ground feature alerts
            if (stairsDetected) {
                feedback.triggerAlert(ALERT_STAIRS_AHEAD);
            } else if (crosswalkDetected) {
                feedback.triggerAlert(ALERT_CROSSWALK);
            } else if (unevenDetected) {
                feedback.triggerAlert(ALERT_UNEVEN_GROUND);
            }

            // Print Serial Diagnostics
            Serial.printf("[FOMO] Infer: %dms (~%.1f FPS) | Road: %d, Sidewalk: %d, Bound: %d\n",
                t_infer, 1000.0 / t_infer, roadCentroids, sidewalkCentroids, rightBoundaryCentroids);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


// ============================================================================
// MAIN SETUP & LOOP
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=======================================================");
    Serial.println("  STANDALONE ASSISTIVE EYEWEAR: FOMO + 3X TOF INFERENCE");
    Serial.println("=======================================================");

    feedback.begin();

    // Initialize 3x VL53L1X Array (SDA: 14, SCL: 15)
    if (!tofArray.begin(14, 15)) {
        Serial.println("[Warning] ToF sensor array failed to initialize!");
    }

    if (!initCamera()) {
        Serial.println("[Error] Camera initialization failed!");
    }

    if (!initTFLM()) {
        Serial.println("[Error] FOMO TFLM initialization failed!");
    }

    // Launch Dual-Core FreeRTOS Tasks
    // Core 0: High-Speed ToF Reflex (Priority 2)
    xTaskCreatePinnedToCore(
        TaskToFReflex,
        "TaskToFReflex",
        4096,
        NULL,
        2,
        &TaskToFHandle,
        0
    );

    // Core 1: FOMO Centroid Perception (Priority 1)
    xTaskCreatePinnedToCore(
        TaskVisionInference,
        "TaskVisionInference",
        8192,
        NULL,
        1,
        &TaskVisionHandle,
        1
    );

    Serial.println("[System] Dual-core FOMO tasks pinned and running!");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
