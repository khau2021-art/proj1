# part_b_calibrate.py
import argparse, os, glob, cv2, numpy as np

def capture(out, w, h, cam=0):
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    i = 0
    print("Space=save, q=quit")
    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.imshow("capture", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            fn = os.path.join(out, f"calib_{i:02d}.png")
            cv2.imwrite(fn, f)
            print("saved", fn)
            i += 1
        elif k in (27, ord('q')):
            break
    cap.release(); cv2.destroyAllWindows()

def calibrate(glob_pat, pattern, square):
    imgs = sorted(glob.glob(glob_pat))
    if not imgs: raise SystemExit("No images found")
    pat = (pattern[0], pattern[1])
    objp = np.zeros((pat[0]*pat[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1,2)
    objp *= square

    objpoints, imgpoints = [], []
    gray = None
    for fn in imgs:
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pat)
        if not ok: 
            print("miss:", fn); continue
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                   (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        objpoints.append(objp); imgpoints.append(corners)
        cv2.drawChessboardCorners(img, pat, corners, ok)
        cv2.imshow("check", img); cv2.waitKey(30)

    if not objpoints: raise SystemExit("No detections collected.")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=0
    )
    print("RMS reprojection error:", ret)
    print("K=\n", K)
    print("dist=", dist.ravel())
    h, w = gray.shape
    np.savez("calibration.npz", K=K, dist=dist, image_size=(w,h),
             pattern_size=pat, square_size=square)
    cv2.destroyAllWindows()
    print("Saved calibration.npz")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capture","calibrate"], required=True)
    ap.add_argument("--out", default="data/calib")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--glob", default="data/calib/calib_*.png")
    ap.add_argument("--pattern", nargs=2, type=int, default=[7,7])
    ap.add_argument("--square", type=float, default=0.020)
    args = ap.parse_args()

    if args.mode == "capture":
        capture(args.out, args.w, args.h)
    else:
        calibrate(args.glob, args.pattern, args.square)

if __name__ == "__main__":
    main()
