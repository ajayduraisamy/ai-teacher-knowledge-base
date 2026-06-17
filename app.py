import cv2
import requests

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    _, buffer = cv2.imencode('.jpg', frame)

    response = requests.post(
        "https://aislyntech.com/Api/27-update.php",
        files={
            "image": (
                "capture.jpg",
                buffer.tobytes(),
                "image/jpeg"
            )
        }
    )

    print(response.text)

cap.release()