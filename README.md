# EchoSight
EchoSight features lightweight ESP32 powered smart glasses equipped with a compact OLED display that converts surrounding sounds into clear visual cues. Built for comfort and everyday use, the glasses help users interpret speech and environmental noise through simple icons, text, and direction indicators. This allows 
individuals with hearing impairments to navigate their surroundings with greater confidence, awareness, and safety.


## Problem Statement
EchoSight addresses the challenges faced by individuals with hearing impairments in perceiving and understanding conversation & environmental sounds. By leveraging advanced audio processing and machine learning techniques, our solution transforms audio cues into visual information displayed through OLED glasses, enhancing spatial awareness and safety for users.

## System Architecture
The system consists of three main components:
1. **Frontend**: React-based web application for user interface and audio capture
2. **Backend**: Python server handling audio processing and sound classification
3. **Hardware**: ESP32-based OLED glasses for visual display

```
User Input → Audio Capture → Sound Processing → Classification → Visual Display
```

## Instructions to Run the App

### Frontend Setup
1. **Clone the repository:**
   ```
   git clone https://github.com/SalinCodes/EchoSight.git
   cd EchoSight
   ```

2. **Install frontend dependencies:**
   ```
   cd frontend
   npm install
   ```

3. **Start the frontend:**
   ```
   npm start
   ```

### Backend Setup
1. **Install Python dependencies:**
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. **Start the backend server:**
   ```
   python app.py
   ```

The application will be available at `http://localhost:3000`

## Tech Stack
- **Frontend**:
  - React with TypeScript
  - Tailwind CSS
  - Web Audio API
  
- **Backend**:
  - Python
  - CLAP (Contrastive Language-Audio Pretraining)
  - Flask
  - Google Speech Recognition API

- **Hardware**:
  - ESP32 Microcontroller
  - OLED Display
  
## Features
- Real-time audio capture and processing
- Environmental sound classification
- Speech-to-text transcription
- Customizable accessibility settings
- Visual sound representation
- Low-latency audio processing
- Intuitive user interface

## Team and Roles
- **Salin Adhikari** - Software Development
- **Larak Yakthumba** - Hardware Integration & ESP32 Programming

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.
