#include <WiFi.h>
#include <SPI.h>
#include <LoRa.h>
#include <FirebaseESP32.h>
#include <HTTPClient.h> // Required for SMS

#define SCK 18
#define MISO 19
#define MOSI 23

// ==========================================
// 1. CREDENTIALS (FILL THESE IN)
// ==========================================
// --- Network Credentials ---
#define WIFI_SSID "xx"
#define WIFI_PASSWORD "xx@xx"

// --- Firebase Credentials ---
#define API_KEY "xxx"
#define DATABASE_URL "xxp"

// --- Twilio SMS Credentials ---
// Get these free at twilio.com
const char* twilio_account_sid = "YOUR_TWILIO_SID";
const char* twilio_auth_token = "YOUR_TWILIO_TOKEN";
const char* twilio_phone_number = "+1234567890"; // Your Twilio Number

// ==========================================
// LORA & FIREBASE SETUP
// ==========================================
#define ss 5
#define rst 14
#define dio0 2

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ==========================================
// DYNAMIC SMS ALERT FUNCTION
// ==========================================
void send_sms_alert(String alert_type, String node_id) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[SMS] WiFi not connected. Aborting.");
        return;
    }

    // 1. Fetch Dynamic Target Phone Number from Firebase (Set by your HTML Dashboard)
    String dynamic_phone = "";
    Serial.println("[FB] Fetching emergency contact number...");
    
    if (Firebase.getString(fbdo, "forest_data/config/alert_phone")) {
        dynamic_phone = fbdo.stringData();
    } else {
        Serial.println("[FB] Failed to get phone number. Aborting SMS.");
        return; 
    }

    if (dynamic_phone.length() < 10) {
        Serial.println("[SMS] Invalid phone number in database.");
        return;
    }

    // 2. Send the HTTP POST Request to Twilio
    HTTPClient http;
    String url = "https://api.twilio.com/2010-04-01/Accounts/" + String(twilio_account_sid) + "/Messages.json";
    http.begin(url);
    
    http.setAuthorization(twilio_account_sid, twilio_auth_token);
    http.addHeader("Content-Type", "application/x-www-form-urlencoded");

    String message = "🚨 FEHARP COMMAND ALERT 🚨\nCritical Event: " + alert_type + "\nSource: " + node_id;
    String post_data = "To=" + dynamic_phone + "&From=" + String(twilio_phone_number) + "&Body=" + message;

    Serial.println("[SMS] Firing alert to " + dynamic_phone + "...");
    int httpResponseCode = http.POST(post_data);

    if (httpResponseCode == 201 || httpResponseCode == 200) {
        Serial.println("[SMS] ✔ Sent Successfully!");
    } else {
        Serial.print("[SMS] ❌ Error HTTP Code: ");
        Serial.println(httpResponseCode);
    }
    http.end();
}

// ==========================================
// MAIN INITIALIZATION
// ==========================================
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

    if (Firebase.signUp(&config, &auth, "", "")) {
        Serial.println("[SYS] Firebase Auth Successful!");
    } else {
        Serial.print("[SYS] Firebase Auth FAILED: ");
        Serial.println(config.signer.signupError.message.c_str());
    }
    Firebase.begin(&config, &auth);
    Firebase.reconnectWiFi(true);

    // 3. Initialize LoRa
    SPI.begin(SCK, MISO, MOSI, ss);
    LoRa.setPins(ss, rst, dio0);
    if (!LoRa.begin(433E6)) { 
        Serial.println("[SYS] LoRa Init Failed!");
        while (1);
    }
    LoRa.setSyncWord(0xF3);
    Serial.println("[SYS] Gateway Ready. Listening for Forest Nodes...");
}

// ==========================================
// CONTINUOUS LISTENING LOOP
// ==========================================
void loop() {
    int packetSize = LoRa.parsePacket();
    
    if (packetSize) {
        String payload = "";
        while (LoRa.available()) {
            payload += (char)LoRa.read();
        }
        
        int rssi = LoRa.packetRssi();
        Serial.println("\n-----------------------------------");
        Serial.println("[LoRa RX] " + payload);
        Serial.println("[RSSI] " + String(rssi) + " dBm");

        // ── 🚨 SMS INTERCEPT LOGIC 🚨 ──
        if (payload.indexOf("GUN_SHOT") > -1) {
            send_sms_alert("Gunshot Detected", "Acoustic Node (N02)");
        } else if (payload.indexOf("CHAINSAW") > -1) {
            send_sms_alert("Illegal Logging (Chainsaw)", "Acoustic Node (N02)");
        } else if (payload.indexOf("FIRE") > -1) {
            send_sms_alert("Fire/High Gas Warning", "Acoustic Node (N02)");
        } else if (payload.indexOf("HUMAN") > -1) {
            send_sms_alert("Human Trespasser", "Vision Node (N01)");
        }

        // ── 🌐 FIREBASE ROUTING LOGIC ──
        if (Firebase.ready()) {
            
            // Route Camera Data
            if (payload.startsWith("CAM|")) {
                if (Firebase.pushString(fbdo, "forest_data/vision_events", payload)) {
                    Serial.println("[FB] Vision Data Pushed OK");
                }
            } 
            // Route Acoustic/Environmental Data
            else if (payload.startsWith("ENV|")) {
                if (Firebase.pushString(fbdo, "forest_data/audio_events", payload)) {
                    Serial.println("[FB] Acoustic Data Pushed OK");
                }
            }
            // Route Pest Data
            else if (payload.startsWith("PEST|")) {
                if (Firebase.pushString(fbdo, "forest_data/pest_events", payload)) {
                    Serial.println("[FB] Pest Data Pushed OK");
                }
            }

        } else {
            Serial.println("[FB] Not ready — skipping push");
        }

        // Force LoRa back into receive mode
        LoRa.receive();
    }
}
