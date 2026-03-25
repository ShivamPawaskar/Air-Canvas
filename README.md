# Air Canvas

Air Canvas is a gesture-driven drawing app built with Python, OpenCV, and MediaPipe. It uses a single locked hand for interaction and provides a cleaner control model for freehand drawing with gesture-based tool switching..

## Live Landing Page

Open the public landing page here:

- [Air Canvas Live Page](https://shivampawaskar.github.io/air-canvas/)

## Features

- Single-hand scan and lock before starting
- Full-screen drawing with webcam hand tracking
- Gesture-based drawing and hovering
- Gesture shortcuts for brush, eraser, colour cycling, and clear-all
- Transparent in-app overlay UI
- Static landing page included with HTML/CSS

## Gesture Controls

- `Index finger` -> Draw
- `Index + middle` -> Hover / interact with toolbar
- `Thumb only` -> Switch to brush
- `Fist` -> Switch to eraser
- `Three fingers` -> Next colour
- `Open palm (hold)` -> Clear canvas

Gesture actions use a short dwell before activation and then a cooldown to reduce accidental triggers.

## Requirements

- Python 3.10+
- Webcam
- Windows, macOS, or Linux with camera access

Python packages:

- `mediapipe`
- `opencv-python`
- `numpy`

## Installation

```powershell
pip install mediapipe opencv-python numpy
```

## Run The App

From the project directory:

```powershell
python air_canvas.py
```

On first launch, the app downloads the MediaPipe hand landmark model if it is not already present.

## Landing Page

A static landing page is included:

- [landing.html](./landing.html)
- [styles.css](./styles.css)
- Public link: [https://shivampawaskar.github.io/air-canvas/](https://shivampawaskar.github.io/air-canvas/)

Open it directly:

```powershell
start landing.html
```

Or serve it locally:

```powershell
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/landing.html
```

## Project Structure

```text
Air Canvas/
├── air_canvas.py
├── landing.html
├── styles.css
├── hand_landmarker.task
└── README.md
```

## Notes

- The runtime application is the Python/OpenCV app in `air_canvas.py`.
- The HTML/CSS files are a separate presentation page, not the live drawing runtime.
- If the webcam opens but gestures feel inconsistent, adjust lighting, camera framing, and hand distance from the camera.

## Future Improvements

- Export and session management
- Adjustable gesture sensitivity
- Multiple UI themes
- Browser-based runtime version

## License

Add a license file if you want to publish this publicly with explicit usage terms.

