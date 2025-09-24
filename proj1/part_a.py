# part_a.py
import argparse, os, cv2, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    win = "Part A (q quit, s save)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    print("Press 's' to save example images; 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # color spaces
        bgr = frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # show
        vis = bgr.copy()
        cv2.putText(vis, "Move mouse and press 's' to save PNG/JPG; 'q' quits.",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('s'):
            cv2.imwrite(os.path.join(args.out, "partA_rgb.png"), bgr)      # PNG (lossless)
            cv2.imwrite(os.path.join(args.out, "partA_rgb.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])  # JPG (lossy)
            cv2.imwrite(os.path.join(args.out, "partA_gray.png"), gray)
            cv2.imwrite(os.path.join(args.out, "partA_hsv.png"), hsv)
            print("Saved: rgb.png, rgb.jpg, gray.png, hsv.png")
            # pixel example at image center:
            h,w = bgr.shape[:2]
            cx, cy = w//2, h//2
            print(f"At center (x={cx},y={cy}):")
            print("  BGR:", bgr[cy, cx].tolist())
            print("  Gray:", int(gray[cy, cx]))
            print("  HSV:", hsv[cy, cx].tolist())

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
