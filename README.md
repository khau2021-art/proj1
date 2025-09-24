Project 1 - Calibration and Localization (CPET-347)
Full Pipeline: Capture > Undistort > Detect Tags > Estimate Pose
Environment: Anaconda (Python 3.11), OpenCV, pupil_apriltags, transforms3d

0) Setup
Create and activate "conda env" by typing in the following in Anaconda:
conda create -n cpet347 python=3.11 -y
conda activate cpet347

Install dependencies using Anaconda:
pip install opencv-python pupil-apriltags transforms3d numpy matplotlib

Folder layout should like this in Visual Studio or File Explorer:
proj1/
    data/
        calib/
        world/
    part_a.py
    part_b_calibrate.py
    part_c_localize.py
    part_d_integrated.py
    README.md

To create this folder layout, run these scripts (in order) in Anaconda:
cd C:\Users\minec\Documents\<folderName>
mkdir proj1
cd proj1
mkdir data
mkdir data\calib
mkdir data\world
type nul > part_a.py
type nul > part_b_calibrate.py
type nul > part_c_localize.py
type nul > part_d_integrated.py
type nul > README.md

You can open up Visual Studio Code by typing in:
code .


1) Part A - Camera and Image Basics
Goal: Demonstrate capture, formats, and color spaces (RGB, Grayscale, HSV)

In Anaconda, type in:
python part_a.py --camera 0 --out data

Press "s" once to save images, "q" to quit.

You will obtain four difference outputs:
"partA_rgb.png" (PNG, lossless, larger file size)
"partA_rgb.jpg" (JPG, lossy, smaller file size)
"partA_gray.png" (Useful for detection/calibration when color isn't needed)
"partA_rsv.png" (Useful for color robustness in dynamic lighting environments)

2) Part B - Intrinsic Calibration
Board: 9x6 Inner Corners, Square Size = 24mm (0.024 m)
Recommendation Resolution: 640x480 (Use same resolution in later parts)

Capture 15-20+ images for calibration by using this script in Anaconda:
python part_b_calibrate.py --mode capture --out data/calib --w 640 --h 480

Press "spacebar" to save each view, "q" to finish.

Calibrate using Zhang's method by using this script in Anaconda:
python part_b_calibrate.py --mode calibrate --glob "data/calib/calib_*.png" --pattern 9 6 --square 0.024

Output: Prints RMS reprojection error, K, dist, and saves "calibration.npz".

3) Part C - Localization with AprilTags
Tag Family: tag36h11
Tag Edge: 66.5mm (printed at 25% scale on 11" x 8.5" print paper)
Neighbor Spacing (Center-To-Center): 216.5mm (150 (edge-to-edge) + 66.5)

Run this script in Anaconda:
python part_c_localize.py --cal calibration.npz --tag-size-mm 66.5 --spacing-mm 216.5

- This will scale K to live camera size if necessary
- Uses the detector's built-in pose (pose_R, pose_t)
- Creates a 2x2 world (IDs 5-8) on Z=0
- Overlays inter-tag distances (should show 216.5mm neighbors, 306.2mm diagonals)

Left-click two points to measure distance on Z=0, Right-click clears, "s" saves snapshot.

4) Part D - Integrated Pipeline

Run this script in Anaconda:
python part_d_integrated.py --cal calibration.npz --tag-size-mm 66.5 --spacing-mm 216.5

- Capture -> Undistort -> Detect -> Pose
- Tag axes overlay, inter-tag distances, fused camera pose
- Click-to-measure on Z=0 like Part C
- Press "s" to save snapshot which appends pose to data/world/poses.csv