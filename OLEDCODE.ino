#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WebServer.h>

// OLED configuration
#define SCREEN_WIDTH 128  
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// WiFi credentials
const char* ssid = "HeHeHe_2.4_plus";
const char* password = "Juttakhane#1234*";

// Web server on port 80
WebServer server(80);

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // OLED Init
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED init failed!"));
    while (true); // Hang
  }

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Connecting to WiFi...");
  display.display();

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("WiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("WiFi Connected");
  display.print("IP: ");
  display.println(WiFi.localIP());
  display.display();

  // Define /display endpoint
  server.on("/display", HTTP_POST, []() {
    if (!server.hasArg("plain")) {
      Serial.println("No message received from backend.");
      server.send(400, "text/plain", "No message body received");

      display.clearDisplay();
      display.setCursor(0, 0);
      display.println("No message 🙁");
      display.display();
      return;
    }

    String message = server.arg("plain");
    Serial.println("✅ Message received from Flask backend:");
    Serial.println(message);

    // Display message
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    display.println("Transcription:");
    int y = 10;
    while (message.length() > 0 && y < SCREEN_HEIGHT) {
      int len = message.length() > 20 ? 20 : message.length();
      String line = message.substring(0, len);
      message = message.substring(len);
      display.setCursor(0, y);
      display.println(line);
      y += 10;
    }

    display.display();
    server.send(200, "text/plain", "Message displayed on OLED!");
  });

  server.begin();
  Serial.println("ESP32 WebServer started.");
}

void loop() {
  server.handleClient();
}