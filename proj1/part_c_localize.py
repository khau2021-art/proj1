# part_c_localize.py
import argparse, os, time, itertools
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
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Part C — AprilTag localization using detector pose")
    ap.add_argument("--cal", required=True, help="calibration.npz from Part B")
    ap.add_argument("--tag-size-mm", type=float, required=True, help="Printed black-square edge (mm)")
    ap.add_argument("--spacing-mm",  type=float, required=True, help="Center-to-center neighbor spacing (mm)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--family", default="tag36h11")
    args = ap.parse_args()

    # Load calibration and open camera
    cal = np.load(args.cal)
    K_cal, dist, imsz = cal["K"], cal["dist"], cal["image_size"]
    cap = cv2.VideoCapture(args.camera)
    ok, f0 = cap.read()
    if not ok: raise SystemExit("Camera read failed")
    live = (f0.shape[1], f0.shape[0])

    # Scale intrinsics to live resolution
    K = scale_K(K_cal, imsz, live)
    fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # AprilTag detector (we’ll use its built-in pose)
    detector = Detector(
        families=args.family, nthreads=2,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=True, decode_sharpening=0.25, debug=False
    )

    TAG_SIZE_M = args.tag_size_mm / 1000.0
    spacing_m  = args.spacing_mm  / 1000.0

    # World (centers) — tags 5–8 on a 2×2 grid on Z=0
    def Txy(x,y):
        T = np.eye(4); T[:3,3] = [x,y,0]; return T
    WORLD = {
        5: Txy(0,0),
        6: Txy(spacing_m,0),
        7: Txy(0,spacing_m),
        8: Txy(spacing_m,spacing_m),
    }

    # UI
    win = "Part C (Left: pick P1,P2 on Z=0; Right: clear; S save; Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    clicks = []

    def on_mouse(e,x,y,flags,param):
        nonlocal clicks
        if e==cv2.EVENT_LBUTTONDOWN:
            (clicks.append((x,y)) if len(clicks)<2 else clicks.__setitem__(1,(x,y)))
        elif e==cv2.EVENT_RBUTTONDOWN:
            clicks = []
    cv2.setMouseCallback(win, on_mouse)

    os.makedirs("data/world", exist_ok=True)
    print(f"[CAL] calib size={tuple(imsz)}, live={live}")
    print(f"[K] fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use detector’s pose (estimate_tag_pose=True)
        dets = detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(fx,fy,cx,cy), tag_size=TAG_SIZE_M
        )

        vis = frame.copy()
        Tlist = []
        tcam = {}  # per-tag translation in CAMERA frame (from detector)

        # draw and collect poses
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

            # axes at the tag
            try:
                cv2.drawFrameAxes(vis, K, dist, d.pose_R, d.pose_t, length=TAG_SIZE_M*0.6, thickness=2)
            except Exception:
                pass

            tcam[d.tag_id] = d.pose_t.reshape(3)
            T_cam_tag = rT(d.pose_R, d.pose_t)
            T_world_cam = WORLD[d.tag_id] @ T_inv(T_cam_tag)
            Tlist.append(T_world_cam)

        # inter-tag distances in CAMERA frame (sanity check vs spacing)
        y = 25
        diag_expected = np.sqrt(2.0) * spacing_m * 1000.0
        cv2.putText(vis, f"Expected neighbor ~ {args.spacing_mm:.1f} mm, diag ~ {diag_expected:.1f} mm",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2); y+=22
        for (i,j) in itertools.combinations(sorted(tcam.keys()),2):
            dmm = np.linalg.norm(tcam[i]-tcam[j]) * 1000.0
            cv2.putText(vis, f"d({i},{j}) ~ {dmm:.1f} mm", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,200,255), 2); y+=20

        # fused camera pose (world <- cam)
        if Tlist:
            Twc = avg_SE3(Tlist)
            t = Twc[:3,3]
            cv2.putText(vis, f"t(m)=[{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}]",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y+=22
        else:
            Twc = None
            cv2.putText(vis, "No known tags", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2); y+=22

        # click-to-click measurement on Z=0
        for k,(u,v) in enumerate(clicks):
            cv2.circle(vis,(u,v),5,(0,0,255),-1)
            cv2.putText(vis,f"P{k+1}",(u+6,v-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        if Twc is not None and len(clicks)==2:
            (u1,v1),(u2,v2) = clicks
            C,d1 = ray_world_from_pixel(u1,v1,Twc,K); P1 = intersect_Z0(C,d1)
            C,d2 = ray_world_from_pixel(u2,v2,Twc,K); P2 = intersect_Z0(C,d2)
            if P1 is not None and P2 is not None:
                mm = np.linalg.norm(P1-P2)*1000.0
                mid = ((u1+u2)//2,(v1+v2)//2)
                cv2.line(vis,(u1,v1),(u2,v2),(255,0,0),2)
                cv2.putText(vis,f"{mm:.1f} mm (Z=0)", (mid[0]+8, mid[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        if k == ord('s') and Twc is not None:
            fn = os.path.join("data/world", f"snap_{int(time.time())}.png")
            cv2.imwrite(fn, frame); print("saved", fn)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
