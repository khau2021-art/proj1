# part_d_integrated.py
import argparse, os, csv, time, itertools
import numpy as np, cv2, transforms3d as t3d
from pupil_apriltags import Detector

# ---------- math helpers ----------
def scale_K(K, calib_size, frame_size):
    cw,ch = map(float, calib_size); fw,fh = map(float, frame_size)
    sx, sy = fw/cw, fh/ch
    K2 = K.copy()
    K2[0,0]*=sx; K2[1,1]*=sy; K2[0,2]*=sx; K2[1,2]*=sy
    return K2

def T_inv(T):
    R,t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3] = -R.T @ t
    return Ti

def rT(R,t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t.reshape(3); return T

def avg_SE3(Ts):
    if len(Ts)==1: return Ts[0]
    t = np.mean([T[:3,3] for T in Ts], axis=0)
    qs=[]
    for T in Ts:
        q = t3d.quaternions.mat2quat(T[:3,:3])
        if qs and np.dot(qs[0], q) < 0: q = -q
        qs.append(q)
    q = np.mean(qs, axis=0); q /= np.linalg.norm(q)
    R = t3d.quaternions.quat2mat(q)
    O = np.eye(4); O[:3,:3]=R; O[:3,3]=t; return O

def ray_world_from_pixel(u,v,T_wc,K):
    Kinv = np.linalg.inv(K)
    d_cam = Kinv @ np.array([u,v,1.0]); d_cam /= np.linalg.norm(d_cam)
    R = T_wc[:3,:3]; C = T_wc[:3,3]
    d = R @ d_cam; d /= np.linalg.norm(d)
    return C,d

def intersect_Z0(C,d):
    if abs(d[2]) < 1e-9: return None
    lam = -C[2]/d[2]
    return C + lam*d

def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Part D — Full pipeline: Capture → Undistort → Detect → Pose")
    ap.add_argument("--cal", required=True, help="calibration.npz from Part B")
    ap.add_argument("--tag-size-mm", type=float, required=True, help="Printed black-square edge (mm)")
    ap.add_argument("--spacing-mm",  type=float, required=True, help="Center-to-center neighbor spacing (mm)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--out", default="data/world", help="Where snapshots and poses.csv are saved")
    ap.add_argument("--no-undistort", action="store_true", help="Disable undistortion (diagnostic)")
    args = ap.parse_args()

    # Load calibration
    cal = np.load(args.cal)
    K_cal, dist, imsz = cal["K"], cal["dist"], cal["image_size"]

    # Camera
    cap = cv2.VideoCapture(args.camera)
    ok, f0 = cap.read()
    if not ok: raise SystemExit("Camera read failed")
    live = (f0.shape[1], f0.shape[0])

    # Intrinsics scaled to live res
    K = scale_K(K_cal, imsz, live)
    fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Undistort maps
    if not args.no_undistort:
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (live[0], live[1]), cv2.CV_16SC2)
    else:
        map1 = map2 = None

    # AprilTag detector
    detector = Detector(
        families=args.family, nthreads=2,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=True, decode_sharpening=0.25, debug=False
    )

    TAG_SIZE_M = args.tag_size_mm / 1000.0
    spacing_m  = args.spacing_mm  / 1000.0

    # World tag centers (IDs 5..8)
    def Txy(x,y):
        T = np.eye(4); T[:3,3] = [x,y,0]; return T
    WORLD = {
        5: Txy(0,0),
        6: Txy(spacing_m,0),
        7: Txy(0,spacing_m),
        8: Txy(spacing_m,spacing_m),
    }

    # Output setup
    os.makedirs(args.out, exist_ok=True)
    poses_csv = os.path.join(args.out, "poses.csv")
    if not os.path.exists(poses_csv):
        with open(poses_csv, "w", newline="") as f:
            csv.writer(f).writerow(["image","tx","ty","tz","yaw","pitch","roll"])

    # UI
    win = "Part D Integrated (Left: pick P1,P2 | Right: clear | S: save | Q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    clicks=[]
    def on_mouse(e,x,y,flags,param):
        nonlocal clicks
        if e==cv2.EVENT_LBUTTONDOWN:
            (clicks.append((x,y)) if len(clicks)<2 else clicks.__setitem__(1,(x,y)))
        elif e==cv2.EVENT_RBUTTONDOWN:
            clicks=[]
    cv2.setMouseCallback(win, on_mouse)

    # One-time banner
    print("Pipeline stages initialized:")
    print("  ✓ Capture Frame")
    print(f"  ✓ Undistort Image ({'disabled' if args.no_undistort else 'enabled'})")
    print("  ✓ Detect Tags (pupil_apriltags, pose from detector)")
    print("  ✓ Estimate Pose (fused world-camera)")

    frames, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break

        if not args.no_undistort:
            und = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            und = frame
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=(fx,fy,cx,cy), tag_size=TAG_SIZE_M)

        vis = und.copy()
        Tlist = []
        tcam = {}

        for d in dets:
            pts = d.corners.astype(int)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,255), 2)
            c = d.center.astype(int)
            cv2.circle(vis, tuple(c), 4, (0,0,255), -1)
            cv2.putText(vis, f"id={d.tag_id}", (c[0]+6,c[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            if (d.pose_R is None) or (d.tag_id not in WORLD):
                continue

            try:
                cv2.drawFrameAxes(vis, K, dist, d.pose_R, d.pose_t,
                                  length=TAG_SIZE_M*0.6, thickness=2)
            except Exception:
                pass

            tcam[d.tag_id] = d.pose_t.reshape(3)
            T_cam_tag = rT(d.pose_R, d.pose_t)
            T_world_cam = WORLD[d.tag_id] @ T_inv(T_cam_tag)
            Tlist.append(T_world_cam)

        # Expected distances overlay
        y = 22
        diag_expected = np.sqrt(2.0) * spacing_m * 1000.0
        cv2.putText(vis, f"Expected neighbor ~ {args.spacing_mm:.1f} mm, diag ~ {diag_expected:.1f} mm",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2); y+=22

        # Inter-tag distances
        for (i,j) in itertools.combinations(sorted(tcam.keys()),2):
            dmm = np.linalg.norm(tcam[i]-tcam[j]) * 1000.0
            cv2.putText(vis, f"d({i},{j}) ~ {dmm:.1f} mm", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,200,255), 2); y+=20

        # Fused camera pose
        if Tlist:
            Twc = avg_SE3(Tlist)
            t = Twc[:3,3]
            yaw,pitch,roll = t3d.euler.mat2euler(Twc[:3,:3], axes='sxyz')
            cv2.putText(vis, f"t(m)=[{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}]  y/p/r={yaw:.2f},{pitch:.2f},{roll:.2f}",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y+=22
        else:
            Twc = None
            cv2.putText(vis, "No known tags", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2); y+=22

        # Z=0 measurement
        for k,(u,v) in enumerate(clicks):
            cv2.circle(vis,(u,v),5,(0,0,255),-1)
            cv2.putText(vis,f"P{k+1}",(u+6,v-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        if Twc is not None and len(clicks)==2:
            (u1,v1),(u2,v2) = clicks
            C,d1 = ray_world_from_pixel(u1,v1,Twc,K); P1 = intersect_Z0(C,d1)
            C,d2 = ray_world_from_pixel(u2,v2,Twc,K); P2 = intersect_Z0(C,d2)
            if P1 is not None and P2 is not None:
                mm = np.linalg.norm(P1-P2) * 1000.0
                mid = ((u1+u2)//2,(v1+v2)//2)
                cv2.line(vis,(u1,v1),(u2,v2),(255,0,0),2)
                cv2.putText(vis,f"{mm:.1f} mm (Z=0)", (mid[0]+8, mid[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        # FPS
        frames += 1
        if frames == 1: t0 = time.time()
        if frames >= 10:
            fps = frames / (time.time() - t0)
            cv2.putText(vis, f"{fps:.1f} FPS", (10, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,50), 2)

        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('s') and Twc is not None:
            name = f"snap_{now_stamp()}.png"
            path = os.path.join(args.out, name)
            cv2.imwrite(path, und)
            t = Twc[:3,3]; yaw,pitch,roll = t3d.euler.mat2euler(Twc[:3,:3], axes='sxyz')
            with open(poses_csv, "a", newline="") as f:
                csv.writer(f).writerow([name, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                                        f"{yaw:.6f}", f"{pitch:.6f}", f"{roll:.6f}"])
            print(f"[SAVE] {path} with pose (m): {t}  y/p/r(rad): {yaw:.2f},{pitch:.2f},{roll:.2f}")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
