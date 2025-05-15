"""
Stereo Camera Calibration

This script provides functionality for calibrating stereo camera systems.
It handles intrinsic and extrinsic parameter calibration for stereo vision
applications using checkerboard pattern detection.

TODO:
- Add error handling for camera connection failures
- Implement automatic checkerboard detection parameter tuning
- Add validation metrics for calibration quality
- Add option for stereocamera that uses the same camera stream
"""

import glob
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy import linalg

# Dictionary to store calibration settings from YAML configuration file
calibration_settings: Dict[str, Any] = {}


@dataclass
class CalibrationState:
    """Class to track and store calibration state between sessions."""

    # Step flags
    frames_collected_cam0: bool = False
    frames_collected_cam1: bool = False
    intrinsics_calibrated_cam0: bool = False
    intrinsics_calibrated_cam1: bool = False
    stereo_frames_collected: bool = False
    stereo_calibrated: bool = False

    # Camera parameters
    camera_matrix0: Optional[np.ndarray] = None
    dist_coeffs0: Optional[np.ndarray] = None
    camera_matrix1: Optional[np.ndarray] = None
    dist_coeffs1: Optional[np.ndarray] = None

    # Stereo calibration results
    rotation_matrix0: Optional[np.ndarray] = None
    translation_vector0: Optional[np.ndarray] = None
    rotation_matrix1: Optional[np.ndarray] = None
    translation_vector1: Optional[np.ndarray] = None

    # Projection matrices
    projection_matrix0: Optional[np.ndarray] = None
    projection_matrix1: Optional[np.ndarray] = None

    # Record when last modified
    last_update: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    # Default filename for state storage
    STATE_FILENAME: ClassVar[str] = "calibration_state.yaml"

    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.last_update = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary suitable for YAML serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the calibration state
        """
        # Start with a standard dataclass dict conversion
        data = asdict(self)

        # Convert numpy arrays to lists for YAML serialization
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationState":
        """
        Create a CalibrationState instance from a dictionary.

        Parameters:
            data: Dictionary containing calibration state data

        Returns:
            CalibrationState: A new calibration state instance
        """
        # Create a copy to avoid modifying the input
        data_copy = data.copy()

        # Convert lists back to numpy arrays where needed
        for key in [
            "camera_matrix0",
            "dist_coeffs0",
            "camera_matrix1",
            "dist_coeffs1",
            "rotation_matrix0",
            "rotation_matrix1",
            "translation_vector0",
            "translation_vector1",
            "projection_matrix0",
            "projection_matrix1",
        ]:
            if key in data_copy and data_copy[key] is not None:
                data_copy[key] = np.array(data_copy[key])

        return cls(**data_copy)

    def save(self, directory: str = "camera_parameters") -> None:
        """
        Save calibration state to a YAML file.

        Parameters:
            directory: Directory to save the state file
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.update_timestamp()

        # Convert to serializable format
        data = self.to_dict()

        # Save to YAML file
        filepath = os.path.join(directory, self.STATE_FILENAME)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"Calibration state saved to {filepath}")

    @classmethod
    def load(cls, directory: str = "camera_parameters") -> Optional["CalibrationState"]:
        """
        Load calibration state from a YAML file if it exists.

        Parameters:
            directory: Directory where state file is located

        Returns:
            CalibrationState object if file exists, None otherwise
        """
        filepath = os.path.join(directory, cls.STATE_FILENAME)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)

            state = cls.from_dict(data)
            print(
                f"Loaded calibration state from {filepath} (last updated: {state.last_update})"
            )
            return state
        except (yaml.YAMLError, KeyError) as e:
            print(f"Error: Could not load calibration state from {filepath}: {str(e)}")
            return None


def print_step_banner(step_number: int, step_description: str) -> None:
    """
    Print a banner to indicate the current calibration step.

    Parameters:
        step_number: Current step number
        step_description: Description of the current step
    """
    print("\n" + "=" * 80)
    print(f"STEP {step_number}: {step_description}")
    print("=" * 80 + "\n")


def triangulate_point(
    projection_matrix1: np.ndarray,
    projection_matrix2: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate a 3D point from two 2D projections using Direct Linear Transform (DLT).

    Parameters:
        projection_matrix1: 3x4 projection matrix of the first camera
        projection_matrix2: 3x4 projection matrix of the second camera
        point1: 2D point in the first camera view (x, y)
        point2: 2D point in the second camera view (x, y)

    Returns:
        np.ndarray: 3D triangulated point (x, y, z)
    """
    # Create coefficient matrix for DLT
    coefficient_matrix = [
        point1[1] * projection_matrix1[2, :] - projection_matrix1[1, :],
        projection_matrix1[0, :] - point1[0] * projection_matrix1[2, :],
        point2[1] * projection_matrix2[2, :] - projection_matrix2[1, :],
        projection_matrix2[0, :] - point2[0] * projection_matrix2[2, :],
    ]
    coefficient_matrix = np.array(coefficient_matrix).reshape((4, 4))

    # Solve using SVD
    btb_matrix = coefficient_matrix.transpose() @ coefficient_matrix
    _, _, v_transpose = linalg.svd(btb_matrix, full_matrices=False)

    # Extract 3D point from the last row of V transpose, normalized
    return v_transpose[3, 0:3] / v_transpose[3, 3]


def parse_calibration_settings_file(filename: str) -> None:
    """
    Parse and load calibration settings from a YAML configuration file.

    Parameters:
        filename: Path to the calibration settings YAML file

    Returns:
        None. Settings are stored in the global calibration_settings dictionary

    Raises:
        SystemExit: If file doesn't exist or doesn't contain expected format
    """
    global calibration_settings

    if not os.path.exists(filename):
        print("Error: File does not exist:", filename)
        sys.exit(1)

    print("Using calibration settings from:", filename)

    with open(filename, "r") as config_file:
        calibration_settings = yaml.safe_load(config_file)

    # Validate that the configuration contains required settings
    if "camera0" not in calibration_settings:
        print(
            "Error: 'camera0' key not found in the settings file. "
            "Check if the correct calibration_settings.yaml file was provided."
        )
        sys.exit(1)


def save_frames_single_camera(camera_name: str) -> None:
    """
    Open camera stream and save frames for calibration.

    This function captures frames from a single camera for intrinsic calibration.

    Parameters:
        camera_name: Name of the camera as defined in the settings file (e.g., "camera0")

    Returns:
        None. Frames are saved to the 'frames' directory with naming pattern {camera_name}_{frame_number}.png
    """
    print(f"Starting frame collection for {camera_name}...")

    # Create frames directory if it doesn't exist
    if not os.path.exists("frames"):
        os.mkdir("frames")

    # Get settings from configuration
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]

    # Open video stream and set resolution
    # Note: If unsupported resolution is used, this does NOT raise an error
    cap = cv2.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(
            f"Error: Could not open camera {camera_name} (device ID: {camera_device_id})"
        )
        return

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No video data received from camera. Exiting...")
            cap.release()
            return

        # Create a smaller version of the frame for display
        frame_small = cv2.resize(frame, None, fx=1 / view_resize, fy=1 / view_resize)

        if not start:
            cv2.putText(
                frame_small,
                "Press SPACEBAR to start collection frames",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv2.putText(
                frame_small,
                f"Cooldown: {cooldown}",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame_small,
                f"Num frames: {saved_count}/{number_to_save}",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # Save the frame when cooldown reaches 0
            if cooldown <= 0:
                save_path = os.path.join("frames", f"{camera_name}_{saved_count}.png")
                cv2.imwrite(save_path, frame)
                saved_count += 1
                print(
                    f"Saved frame {saved_count}/{number_to_save} for {camera_name}",
                    end="\r",
                )
                cooldown = cooldown_time

        cv2.imshow("frame_small", frame_small)
        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            print("Frame collection cancelled.")
            break

        if key == 32:  # SPACE key
            start = True

        # Break out of the loop when enough frames have been saved
        if saved_count == number_to_save:
            print(f"Completed collecting {number_to_save} frames for {camera_name}")
            break

    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera_for_intrinsic_parameters(
    images_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate a single camera to obtain its intrinsic parameters.

    This function processes images containing checkerboard patterns to calculate
    the camera matrix and distortion coefficients.

    Parameters:
        images_prefix (str): Path prefix for the calibration images (e.g., "frames/camera0*")

    Returns:
        tuple: Camera matrix and distortion coefficients
            - camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
            - distortion_coeffs (numpy.ndarray): Camera distortion coefficients
    """
    print(f"Calibrating {images_prefix.split('/')[-1]} intrinsics...")
    # Find all images matching the prefix pattern
    image_paths = sorted(glob.glob(images_prefix))

    if not image_paths:
        print(f"Error: No images found matching {images_prefix}")
        return None, None

    print(f"Found {len(image_paths)} calibration images")

    # Read all calibration frames
    images = [cv2.imread(image_path, 1) for image_path in image_paths]
    total_frames = len(images)

    # Criteria used by checkerboard pattern detector
    # (type, max_iterations, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # Define 3D coordinates of checkerboard corners in the checkerboard coordinate system
    checkerboard_points_3d = np.zeros((rows * columns, 3), np.float32)
    checkerboard_points_3d[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    checkerboard_points_3d = world_scaling * checkerboard_points_3d

    # Get frame dimensions (assuming all frames have the same size)
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of detected checkerboard corners
    image_points = []  # 2D points in image plane

    # Corresponding 3D coordinates in world space
    object_points = []  # 3D points in real world space

    for i, frame in enumerate(images):
        print(f"Processing frame {i+1}/{total_frames}", end="\r")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners
        found_checkerboard, corners = cv2.findChessboardCorners(
            gray, (rows, columns), None
        )

        # Draw frame number and progress
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Frame {i+1}/{total_frames}",
            (25, 25),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 0, 255),
            1,
        )

        if found_checkerboard:
            # Refine corner detection
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(
                display_frame, (rows, columns), corners, found_checkerboard
            )
            cv2.putText(
                display_frame,
                'Press "s" to skip, any other key to use this image',
                (25, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 255),
                1,
            )

            cv2.imshow("Calibration Image", display_frame)
            k = cv2.waitKey(0)

            if k & 0xFF == ord("s"):
                print(f"\nSkipping frame {i+1}")
                continue

            object_points.append(checkerboard_points_3d)
            image_points.append(corners)
        else:
            print(f"No checkerboard found in frame {i+1}")
            cv2.putText(
                display_frame,
                "No checkerboard detected! Press any key to continue.",
                (25, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 255),
                1,
            )
            cv2.imshow("Calibration Image", display_frame)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nPerforming camera calibration...")

    if not object_points:
        print("Error: No valid calibration frames detected")
        return None, None

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None
    )
    print("Camera calibration complete")
    print("RMSE:", ret)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coeffs:", distortion_coeffs)

    return camera_matrix, distortion_coeffs


def save_frames_two_cams(camera0_name: str, camera1_name: str) -> None:
    """
    Open both cameras and capture synchronized frames for stereo calibration.

    Parameters:
        camera0_name: Name of the first camera as defined in the settings
        camera1_name: Name of the second camera as defined in the settings

    Returns:
        None. Frame pairs are saved to the 'frames_pair' directory
    """
    print(
        f"Starting synchronized frame collection for {camera0_name} and {camera1_name}..."
    )

    # Create frames directory for pairs if it doesn't exist
    if not os.path.exists("frames_pair"):
        os.mkdir("frames_pair")

    # Get settings for data capture
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]
    number_to_save = calibration_settings["stereo_calibration_frames"]

    # Open video streams for both cameras
    cap0 = cv2.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv2.VideoCapture(calibration_settings[camera1_name])

    # Check if cameras opened successfully
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras")
        if cap0.isOpened():
            cap0.release()
        if cap1.isOpened():
            cap1.release()
        return

    # Set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Cameras not returning video data. Exiting...")
            break

        # Create smaller versions of frames for display
        frame0_small = cv2.resize(
            frame0, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )
        frame1_small = cv2.resize(
            frame1, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )

        if not start:
            cv2.putText(
                frame0_small,
                "Make sure both cameras can see the calibration pattern well",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                frame0_small,
                "Press SPACEBAR to start collection frames",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            # Display cooldown and frame count on both camera views
            for frame in [frame0_small, frame1_small]:
                cv2.putText(
                    frame,
                    f"Cooldown: {cooldown}",
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Num frames: {saved_count}/{number_to_save}",
                    (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )

            # Save the frame pair when cooldown reaches 0
            if cooldown <= 0:
                cv2.imwrite(
                    os.path.join("frames_pair", f"{camera0_name}_{saved_count}.png"),
                    frame0,
                )
                cv2.imwrite(
                    os.path.join("frames_pair", f"{camera1_name}_{saved_count}.png"),
                    frame1,
                )
                saved_count += 1
                print(
                    f"Saved stereo frame pair {saved_count}/{number_to_save}", end="\r"
                )
                cooldown = cooldown_time

        cv2.imshow("Camera 0", frame0_small)
        cv2.imshow("Camera 1", frame1_small)
        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            print("Frame collection cancelled.")
            break

        if key == 32:  # SPACE key
            start = True
            print("Stereo frame collection started...")

        # Break out of the loop when enough frames have been saved
        if saved_count == number_to_save:
            print(f"Completed collecting {number_to_save} stereo frame pairs")
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


def stereo_calibrate(
    mtx0: np.ndarray,
    dist0: np.ndarray,
    mtx1: np.ndarray,
    dist1: np.ndarray,
    frames_prefix_c0: str,
    frames_prefix_c1: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stereo calibration to find transformation between two cameras.

    This uses paired images with visible checkerboard patterns to calculate
    the rotation and translation from camera0 to camera1.

    Parameters:
        mtx0: Intrinsic matrix of the first camera
        dist0: Distortion coefficients of the first camera
        mtx1: Intrinsic matrix of the second camera
        dist1: Distortion coefficients of the second camera
        frames_prefix_c0: Path prefix for the first camera's frames
        frames_prefix_c1: Path prefix for the second camera's frames

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix and translation vector from camera0 to camera1
    """
    print("Performing stereo calibration...")
    # Read the synchronized frame pairs
    c0_images_paths = sorted(glob.glob(frames_prefix_c0))
    c1_images_paths = sorted(glob.glob(frames_prefix_c1))

    if not c0_images_paths or not c1_images_paths:
        print("Error: No stereo image pairs found")
        return None, None

    if len(c0_images_paths) != len(c1_images_paths):
        print(
            f"Warning: Number of frames doesn't match between cameras: {len(c0_images_paths)} vs {len(c1_images_paths)}"
        )

    total_pairs = min(len(c0_images_paths), len(c1_images_paths))
    print(f"Found {total_pairs} stereo image pairs")

    # Load all image pairs
    c0_images = [
        cv2.imread(image_path, 1) for image_path in c0_images_paths[:total_pairs]
    ]
    c1_images = [
        cv2.imread(image_path, 1) for image_path in c1_images_paths[:total_pairs]
    ]

    # Criteria for subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Get calibration pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # Define 3D coordinates of checkerboard corners in world space
    checkerboard_points_3d = np.zeros((rows * columns, 3), np.float32)
    checkerboard_points_3d[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    checkerboard_points_3d = world_scaling * checkerboard_points_3d

    # Get frame dimensions
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    # Store detected checkerboard corner points
    image_points_left = []  # 2D points in camera0 image plane
    image_points_right = []  # 2D points in camera1 image plane
    object_points = []  # Corresponding 3D points in checkerboard space

    # Process each pair of frames
    used_pairs = 0
    for i, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
        print(f"Processing stereo pair {i+1}/{total_pairs}", end="\r")
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners in both images
        ret0, corners0 = cv2.findChessboardCorners(gray0, (rows, columns), None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)

        display_frame0 = frame0.copy()
        display_frame1 = frame1.copy()

        # If checkerboard is found in both images
        if ret0 and ret1:
            # Refine corner locations to subpixel accuracy
            corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

            # Mark the first corner (origin) with an "O"
            p0_c0 = corners0[0, 0].astype(np.int32)
            p0_c1 = corners1[0, 0].astype(np.int32)

            # Draw checkerboard corners and mark origin on both frames
            for frame, corners, ret, origin_point in [
                (display_frame0, corners0, ret0, p0_c0),
                (display_frame1, corners1, ret1, p0_c1),
            ]:
                cv2.putText(
                    frame,
                    "O",
                    (origin_point[0], origin_point[1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                )
                cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)

                # Add progress indicator to frames
                cv2.putText(
                    frame,
                    f"Pair {i+1}/{total_pairs} - Used: {used_pairs}",
                    (25, 25),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    'Press "s" to skip, any other key to use this pair',
                    (25, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

            # Display frames with detected corners
            cv2.imshow("Camera 0", display_frame0)
            cv2.imshow("Camera 1", display_frame1)
            key = cv2.waitKey(0)

            # Skip this pair if 's' is pressed
            if key & 0xFF == ord("s"):
                print(f"\nSkipping stereo pair {i+1}")
                continue

            # Store points for calibration
            object_points.append(checkerboard_points_3d)
            image_points_left.append(corners0)
            image_points_right.append(corners1)
            used_pairs += 1
        else:
            # Show message when checkerboard not found
            if not ret0:
                cv2.putText(
                    display_frame0,
                    "No checkerboard detected!",
                    (25, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                )
            if not ret1:
                cv2.putText(
                    display_frame1,
                    "No checkerboard detected!",
                    (25, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                )

            cv2.imshow("Camera 0", display_frame0)
            cv2.imshow("Camera 1", display_frame1)
            cv2.waitKey(500)  # Show briefly and continue

    cv2.destroyAllWindows()
    print(f"Using {used_pairs}/{total_pairs} stereo pairs for calibration")

    if used_pairs == 0:
        print("Error: No valid stereo pairs found. Cannot perform calibration.")
        return None, None

    # Perform stereo calibration, keeping intrinsics fixed
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    (
        ret,
        _,
        _,
        _,
        _,
        rotation_matrix,
        translation_vector,
        essential_matrix,
        fundamental_matrix,
    ) = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags,
    )

    print("Stereo calibration complete")
    print("Stereo calibration RMSE:", ret)
    return rotation_matrix, translation_vector


def _make_homogeneous_rep_matrix(
    rotation_matrix: np.ndarray, translation_vector: np.ndarray
) -> np.ndarray:
    """
    Convert rotation matrix and translation vector to a 4x4 homogeneous transformation matrix.

    Parameters:
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix
    """
    transform_matrix = np.zeros((4, 4))
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector.reshape(3)
    transform_matrix[3, 3] = 1

    return transform_matrix


def get_projection_matrix(
    camera_matrix: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute the camera projection matrix from intrinsic and extrinsic parameters.

    Parameters:
        camera_matrix: 3x3 camera intrinsic matrix
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector

    Returns:
        np.ndarray: 3x4 camera projection matrix
    """
    # Create homogeneous transformation matrix and extract the 3x4 component
    homogeneous_matrix = _make_homogeneous_rep_matrix(
        rotation_matrix, translation_vector
    )
    projection_matrix = camera_matrix @ homogeneous_matrix[:3, :]
    return projection_matrix


def check_calibration(
    camera0_name: str,
    camera0_data: List[np.ndarray],
    camera1_name: str,
    camera1_data: List[np.ndarray],
    _zshift: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize calibration results by projecting 3D axes onto both camera views.

    Parameters:
        camera0_name: Name of the first camera as defined in settings
        camera0_data: List containing [camera_matrix, distortion, rotation_matrix, translation_vector] for camera0
        camera1_name: Name of the second camera as defined in settings
        camera1_data: List containing [camera_matrix, distortion, rotation_matrix, translation_vector] for camera1
        _zshift: Distance to shift the coordinate axes away from cameras for better visibility

    Returns:
        Tuple[np.ndarray, np.ndarray]: Projection matrices for both cameras
    """
    print("Verifying calibration results...")
    # Extract camera parameters from the data lists
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    # Calculate projection matrices for both cameras
    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    # Define coordinate axes in 3D space (origin, X, Y, Z)
    coordinate_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Shift axes in Z direction for better visibility and scale them
    z_shift = np.array([0.0, 0.0, _zshift]).reshape((1, 3))
    axes_points_3d = 5 * coordinate_points + z_shift

    # Project 3D axes points to each camera view
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for point_3d in axes_points_3d:
        # Convert to homogeneous coordinates
        point_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

        # Project to camera0
        point_projected = P0 @ point_homogeneous
        pixel_coords = (
            np.array([point_projected[0], point_projected[1]]) / point_projected[2]
        )
        pixel_points_camera0.append(pixel_coords)

        # Project to camera1
        point_projected = P1 @ point_homogeneous
        pixel_coords = (
            np.array([point_projected[0], point_projected[1]]) / point_projected[2]
        )
        pixel_points_camera1.append(pixel_coords)

    # Convert lists to numpy arrays for easier handling
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    # Open video streams
    cap0 = cv2.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv2.VideoCapture(calibration_settings[camera1_name])

    # Set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    # Use RGB colors to represent XYZ axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Video stream not returning frame data")
            sys.exit(1)

        # Draw coordinate axes on camera0 view
        origin0 = tuple(pixel_points_camera0[0].astype(np.int32))
        for color, point in zip(colors, pixel_points_camera0[1:]):
            point_pixel = tuple(point.astype(np.int32))
            cv2.line(frame0, origin0, point_pixel, color, 2)

        # Draw coordinate axes on camera1 view
        origin1 = tuple(pixel_points_camera1[0].astype(np.int32))
        for color, point in zip(colors, pixel_points_camera1[1:]):
            point_pixel = tuple(point.astype(np.int32))
            cv2.line(frame1, origin1, point_pixel, color, 2)

        # Display the frames with projected axes
        cv2.imshow("Camera 0", frame0)
        cv2.imshow("Camera 1", frame1)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    return P0, P1


def get_world_space_origin(
    camera_matrix: np.ndarray, distortion_coeffs: np.ndarray, img_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rotation and translation from world space to camera space.

    Uses a reference image with checkerboard pattern to define world origin.

    Parameters:
        camera_matrix: Camera intrinsic matrix
        distortion_coeffs: Camera distortion coefficients
        img_path: Path to reference image with checkerboard

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix and translation vector from world to camera
    """
    # Read the reference image
    frame = cv2.imread(img_path, 1)

    # Get checkerboard pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]

    # Define 3D coordinates of checkerboard corners in world space
    object_points = np.zeros((rows * columns, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    object_points = world_scaling * object_points

    # Find checkerboard corners in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found_checkerboard, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

    # Draw detected corners on the image for visualization
    cv2.drawChessboardCorners(frame, (rows, columns), corners, found_checkerboard)
    cv2.putText(
        frame,
        "If you don't see detected points, try with a different image",
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        1,
    )
    cv2.imshow("Reference Image", frame)
    cv2.waitKey(0)

    # Solve the PnP problem to get rotation and translation
    _, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, distortion_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return rotation_matrix, tvec


def get_cam1_to_world_transforms(
    cmtx0: np.ndarray,
    dist0: np.ndarray,
    R_W0: np.ndarray,
    T_W0: np.ndarray,
    cmtx1: np.ndarray,
    dist1: np.ndarray,
    R_01: np.ndarray,
    T_01: np.ndarray,
    image_path0: str,
    image_path1: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transformation from world space to camera1 space.

    This function computes and visualizes the world-to-camera1 transformation.

    Parameters:
        cmtx0: Intrinsic matrix of camera0
        dist0: Distortion coefficients of camera0
        R_W0: Rotation matrix from world to camera0
        T_W0: Translation vector from world to camera0
        cmtx1: Intrinsic matrix of camera1
        dist1: Distortion coefficients of camera1
        R_01: Rotation matrix from camera0 to camera1
        T_01: Translation vector from camera0 to camera1
        image_path0: Path to reference image from camera0
        image_path1: Path to reference image from camera1

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix and translation vector from world to camera1
    """
    # Load reference images
    frame0 = cv2.imread(image_path0, 1)
    frame1 = cv2.imread(image_path1, 1)

    # Define unit coordinate frame points (origin, X, Y, Z)
    unit_points = 5 * np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
    ).reshape((4, 1, 3))

    # RGB colors to indicate XYZ axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Project axis points onto camera0 view
    points_projected, _ = cv2.projectPoints(unit_points, R_W0, T_W0, cmtx0, dist0)
    points_2d = points_projected.reshape((4, 2)).astype(np.int32)

    # Draw axes on camera0 image
    origin = tuple(points_2d[0])
    for color, point in zip(colors, points_2d[1:]):
        point_pixel = tuple(point.astype(np.int32))
        cv2.line(frame0, origin, point_pixel, color, 2)
    # Compute camera1 to world transformation

    # R_W1 = R_01 * R_W0  (matrix multiplication)
    # T_W1 = R_01 * T_W0 + T_01
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01

    # Project axis points onto camera1 view
    points_projected, _ = cv2.projectPoints(unit_points, R_W1, T_W1, cmtx1, dist1)
    points_2d = points_projected.reshape((4, 2)).astype(np.int32)

    # Draw axes on camera1 image
    origin = tuple(points_2d[0])
    for color, point in zip(colors, points_2d[1:]):
        point_pixel = tuple(point.astype(np.int32))
        cv2.line(frame1, origin, point_pixel, color, 2)

    # Display the visualization
    cv2.imshow("Camera 0 with world axes", frame0)
    cv2.imshow("Camera 1 with world axes", frame1)
    cv2.waitKey(0)

    return R_W1, T_W1


def ask_yes_no_question(question: str) -> bool:
    """
    Ask a yes/no question to the user.

    Parameters:
        question: Question to ask

    Returns:
        bool: True if yes, False if no
    """
    while True:
        response = input(f"{question} (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please answer with 'y' or 'n'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Call with settings filename: "python3 calibrate.py calibration_settings.yaml"'
        )
        sys.exit(1)

    # Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])

    # Try to load existing calibration state
    calibration_state = CalibrationState.load()

    # Initialize state if none exists
    if calibration_state is None:
        calibration_state = CalibrationState()
        print("Starting new calibration process")
    else:
        print("\nResuming from previous calibration session:")
        print(
            f"- Camera 0 frames collected: {'Yes' if calibration_state.frames_collected_cam0 else 'No'}"
        )
        print(
            f"- Camera 1 frames collected: {'Yes' if calibration_state.frames_collected_cam1 else 'No'}"
        )
        print(
            f"- Camera 0 intrinsics calibrated: {'Yes' if calibration_state.intrinsics_calibrated_cam0 else 'No'}"
        )
        print(
            f"- Camera 1 intrinsics calibrated: {'Yes' if calibration_state.intrinsics_calibrated_cam1 else 'No'}"
        )
        print(
            f"- Stereo frames collected: {'Yes' if calibration_state.stereo_frames_collected else 'No'}"
        )
        print(
            f"- Stereo calibrated: {'Yes' if calibration_state.stereo_calibrated else 'No'}"
        )

    """Step1. Save calibration frames for single cameras"""
    print_step_banner(1, "SAVE CALIBRATION FRAMES FOR SINGLE CAMERAS")

    # Camera 0
    if not calibration_state.frames_collected_cam0:
        save_frames_single_camera("camera0")
        calibration_state.frames_collected_cam0 = True
        calibration_state.save()
    else:
        print("Camera 0 frames already collected. Skipping.")

        # Option to recollect if needed
        if ask_yes_no_question("Do you want to recollect frames for camera0?"):
            save_frames_single_camera("camera0")
            calibration_state.frames_collected_cam0 = True
            calibration_state.save()

    # Camera 1
    if not calibration_state.frames_collected_cam1:
        save_frames_single_camera("camera1")
        calibration_state.frames_collected_cam1 = True
        calibration_state.save()
    else:
        print("Camera 1 frames already collected. Skipping.")

        # Option to recollect if needed
        if ask_yes_no_question("Do you want to recollect frames for camera1?"):
            save_frames_single_camera("camera1")
            calibration_state.frames_collected_cam1 = True
            calibration_state.save()

    """Step2. Obtain camera intrinsic matrices and save them"""
    print_step_banner(2, "CALIBRATE CAMERA INTRINSICS")

    # Camera 0 intrinsics
    if not calibration_state.intrinsics_calibrated_cam0:
        images_prefix = os.path.join("frames", "camera0*")
        cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)

        if cmtx0 is not None and dist0 is not None:
            calibration_state.camera_matrix0 = cmtx0
            calibration_state.dist_coeffs0 = dist0
            calibration_state.intrinsics_calibrated_cam0 = True
            calibration_state.save()
    else:
        print("Camera 0 intrinsics already calibrated. Loading parameters...")
        cmtx0 = calibration_state.camera_matrix0
        dist0 = calibration_state.dist_coeffs0

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to recalibrate camera0 intrinsics?"):
            images_prefix = os.path.join("frames", "camera0*")
            cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)

            if cmtx0 is not None and dist0 is not None:
                calibration_state.camera_matrix0 = cmtx0
                calibration_state.dist_coeffs0 = dist0
                calibration_state.intrinsics_calibrated_cam0 = True
                calibration_state.save()

    # Camera 1 intrinsics
    if not calibration_state.intrinsics_calibrated_cam1:
        images_prefix = os.path.join("frames", "camera1*")
        cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)

        if cmtx1 is not None and dist1 is not None:
            calibration_state.camera_matrix1 = cmtx1
            calibration_state.dist_coeffs1 = dist1
            calibration_state.intrinsics_calibrated_cam1 = True
            calibration_state.save()
    else:
        print("Camera 1 intrinsics already calibrated. Loading parameters...")
        cmtx1 = calibration_state.camera_matrix1
        dist1 = calibration_state.dist_coeffs1

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to recalibrate camera1 intrinsics?"):
            images_prefix = os.path.join("frames", "camera1*")
            cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)

            if cmtx1 is not None and dist1 is not None:
                calibration_state.camera_matrix1 = cmtx1
                calibration_state.dist_coeffs1 = dist1
                calibration_state.intrinsics_calibrated_cam1 = True
                calibration_state.save()

    """Step3. Save calibration frames for both cameras simultaneously"""
    print_step_banner(3, "COLLECT SYNCHRONIZED STEREO FRAMES")

    if not calibration_state.stereo_frames_collected:
        save_frames_two_cams("camera0", "camera1")
        calibration_state.stereo_frames_collected = True
        calibration_state.save()
    else:
        print("Stereo frames already collected. Skipping.")

        # Option to recollect if needed
        if ask_yes_no_question("Do you want to recollect stereo frames?"):
            save_frames_two_cams("camera0", "camera1")
            calibration_state.stereo_frames_collected = True
            calibration_state.save()

    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    print_step_banner(4, "PERFORM STEREO CALIBRATION")

    if not calibration_state.stereo_calibrated:
        frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
        frames_prefix_c1 = os.path.join("frames_pair", "camera1*")

        # camera0 rotation and translation is identity matrix and zeros vector
        R0 = np.eye(3, dtype=np.float32)
        T0 = np.array([0.0, 0.0, 0.0]).reshape((3, 1))

        R1, T1 = stereo_calibrate(
            cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
        )

        if R1 is not None and T1 is not None:
            calibration_state.rotation_matrix0 = R0
            calibration_state.translation_vector0 = T0
            calibration_state.rotation_matrix1 = R1
            calibration_state.translation_vector1 = T1
            calibration_state.stereo_calibrated = True
            calibration_state.save()
    else:
        print("Stereo calibration already completed. Loading parameters...")
        R0 = calibration_state.rotation_matrix0
        T0 = calibration_state.translation_vector0
        R1 = calibration_state.rotation_matrix1
        T1 = calibration_state.translation_vector1

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to perform stereo calibration again?"):
            frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
            frames_prefix_c1 = os.path.join("frames_pair", "camera1*")
            R1, T1 = stereo_calibrate(
                cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
            )

            if R1 is not None and T1 is not None:
                calibration_state.rotation_matrix1 = R1
                calibration_state.translation_vector1 = T1
                calibration_state.stereo_calibrated = True
                calibration_state.save()

    """Step5. Save calibration data where camera0 defines the world space origin."""
    print_step_banner(5, "SAVE EXTRINSIC PARAMETERS AND VERIFY CALIBRATION")

    # Check calibration results by visualizing coordinate axes
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    P0, P1 = check_calibration(
        "camera0", camera0_data, "camera1", camera1_data, _zshift=60.0
    )

    # Update calibration state with projection matrices
    calibration_state.projection_matrix0 = P0
    calibration_state.projection_matrix1 = P1
    calibration_state.save()

    """Optional. Define a different origin point and save the calibration data"""
    print_step_banner(6, "OPTIONAL: DEFINE ALTERNATIVE WORLD ORIGIN (OPTIONAL)")

    if ask_yes_no_question("Do you want to define an alternative world origin?"):
        print("Define a different world space origin using a checkerboard...")
        # Get the world to camera0 rotation and translation
        R_W0, T_W0 = get_world_space_origin(
            cmtx0, dist0, os.path.join("frames_pair", "camera0_4.png")
        )
        # Get rotation and translation from world directly to camera1
        R_W1, T_W1 = get_cam1_to_world_transforms(
            cmtx0,
            dist0,
            R_W0,
            T_W0,
            cmtx1,
            dist1,
            R1,
            T1,
            os.path.join("frames_pair", "camera0_4.png"),
            os.path.join("frames_pair", "camera1_4.png"),
        )

        # Calculate projection matrices for world origin
        P_W0 = get_projection_matrix(cmtx0, R_W0, T_W0)
        P_W1 = get_projection_matrix(cmtx1, R_W1, T_W1)

        # Create a special calibration state for the world origin
        world_state = CalibrationState()
        world_state.camera_matrix0 = cmtx0
        world_state.dist_coeffs0 = dist0
        world_state.camera_matrix1 = cmtx1
        world_state.dist_coeffs1 = dist1
        world_state.rotation_matrix0 = R_W0
        world_state.translation_vector0 = T_W0
        world_state.rotation_matrix1 = R_W1  # Store rotation from world to camera1
        world_state.translation_vector1 = (
            T_W1  # Store translation from world to camera1
        )
        world_state.projection_matrix0 = P_W0
        world_state.projection_matrix1 = P_W1
        world_state.intrinsics_calibrated_cam0 = True
        world_state.intrinsics_calibrated_cam1 = True
        world_state.stereo_calibrated = True

        # Save to a different file
        world_state.STATE_FILENAME = "world_calibration_state.yaml"
        world_state.save()

        print("Alternative world origin calibration complete and saved.")
    else:
        print("Skipping alternative world origin definition.")

    print("\nCalibration process complete!")
