
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        required=True,
        help="Video stream URL displayed by IP Webcam",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.url)

    if not cap.isOpened():
        raise SystemExit(
            "Could not open phone camera. Check the stream URL, "
            "Wi-Fi connection, and IP Webcam server."
        )

    print("Connected to phone camera. Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("No frame received. Check the phone stream.")
                break

            cv2.imshow("Samsung M34 - Live Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
