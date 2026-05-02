#include <WiFi.h>
#include <SPI.h>
#include <LoRa.h>
#include <FirebaseESP32.h>
#define SCK 18
#define MISO 19
#define MOSI 23
// --- Network Credentials ---
#define WIFI_SSID "xx"
#define WIFI_PASSWORD "xx@xx"
#define API_KEY "xxx"
#define DATABASE_URL "xxp"
// --- LoRa Pins ---
#define ss 5
#define rst 14
#define dio0 2

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

void setup() {
  Serial.begin(115200);

  // 1. Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(500); }
  Serial.println("[SYS] WiFi Connected.");

  // 2. Initialize Firebase
  Serial.println("[SYS] Attempting Firebase Login...");
  
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  // Try to sign up, and print the exact error if it fails
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("[SYS] Firebase Auth Successful!");
  } else {
    Serial.print("[SYS] Firebase Auth FAILED: ");
    Serial.println(config.signer.signupError.message.c_str());
    // We won't freeze the board here, we'll let it move on to LoRa anyway
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  // 3. Initialize LoRa
  SPI.begin(SCK, MISO, MOSI, ss);
  LoRa.setPins(ss, rst, dio0);
  if (!LoRa.begin(433E6)) { // Change to 868E6 or 915E6 depending on your hardware
    Serial.println("[SYS] LoRa Init Failed!");
    while (1);
  }
  LoRa.setSyncWord(0xF3);
  Serial.println("[SYS] Gateway Ready. Listening...");

}

void loop() {
    // Always listening
    int packetSize = LoRa.parsePacket();

    if (packetSize) {
        String payload = "";
        while (LoRa.available()) {
            payload += (char)LoRa.read();
        }

        int rssi = LoRa.packetRssi();
        Serial.println("[LoRa RX] " + payload);
        Serial.println("[RSSI] " + String(rssi) + " dBm");

        if (payload.startsWith("CAM|")) {
            if (Firebase.ready()) {
                if (Firebase.pushString(fbdo,
                    "forest_data/vision_events", payload)) {
                    Serial.println("[FB] Pushed OK");
                } else {
                    Serial.println("[FB] FAILED: "
                        + fbdo.errorReason());
                }
            } else {
                Serial.println("[FB] Not ready — skipping push");
            }
        }

        // Force LoRa back into continuous receive mode
        LoRa.receive();
    }
}
