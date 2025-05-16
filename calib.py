"""
Stereo Camera Calibration

This script provides functionality for calibrating stereo camera systems.
It handles intrinsic and extrinsic parameter calibration for stereo vision
applications using checkerboard pattern detection.
"""

import glob
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from utils import (
    CalibrationState,
    crop_frame,
    parse_calibration_settings_file,
    split_stereo_frame,
)

# Global variable to store calibration settings
calibration_settings: Dict[str, Any] = None


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

    # Check if we're in single camera stereo mode
    use_single_camera = calibration_settings.get("use_single_camera_for_stereo", False)

    # If we're in single camera stereo mode and this is camera1, we can skip
    # as we'll capture both frames when processing camera0
    if use_single_camera and camera_name == "camera1":
        print(
            "Using single camera for stereo - camera1 frames will be captured with camera0"
        )
        return

    # Get settings from configuration
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]
    crop_percentage = calibration_settings.get("crop_percentage", 0)

    if crop_percentage > 0:
        print(f"Camera {camera_name} will be cropped by {crop_percentage}%")

    if use_single_camera:
        print(f"Using single camera stereo mode with device ID: {camera_device_id}")
        # Create frames directory for pairs if it doesn't exist
        if not os.path.exists("frames_pair"):
            os.mkdir("frames_pair")

    # Open video stream and set resolution
    # Note: If unsupported resolution is used, this does NOT raise an error
    cap = cv2.VideoCapture(camera_device_id)

    # If using single camera mode, we need to double the width
    if use_single_camera:
        # Set resolution to double width for side-by-side stereo
        cap.set(3, width * 2)
        cap.set(4, height)
    else:
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

        # If we're in single camera stereo mode, split the frame
        if use_single_camera:
            left_frame, right_frame = split_stereo_frame(frame)

            # Use appropriate half based on which camera we're collecting for
            if camera_name == "camera0":
                frame = left_frame

        # Apply cropping if needed
        if crop_percentage > 0:
            frame = crop_frame(frame, crop_percentage)
            if use_single_camera:
                right_frame = crop_frame(right_frame, crop_percentage)

        # Create a smaller version of the frame for display
        frame_small = cv2.resize(frame, None, fx=1 / view_resize, fy=1 / view_resize)

        if not start:
            cv2.putText(
                frame_small,
                "Press SPACEBAR to start collection frames",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
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
                0.7,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame_small,
                f"Num frames: {saved_count}/{number_to_save}",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                1,
            )

            # Save the frame when cooldown reaches 0
            if cooldown <= 0:
                save_path = os.path.join("frames", f"{camera_name}_{saved_count}.png")
                cv2.imwrite(save_path, frame)

                # If we're using single camera mode and this is camera0,
                # we should also save camera1 at the same time
                if use_single_camera and camera_name == "camera0":
                    cam1_path = os.path.join("frames", f"camera1_{saved_count}.png")
                    cv2.imwrite(cam1_path, right_frame)
                    cv2.imwrite(
                        os.path.join("frames_pair", f"camera0_{saved_count}.png"),
                        frame,
                    )
                    cv2.imwrite(
                        os.path.join("frames_pair", f"camera1_{saved_count}.png"),
                        right_frame,
                    )
                    print(
                        f"Saved both camera0 and camera1 frames {saved_count}/{number_to_save}",
                        end="\r",
                    )
                else:
                    print(
                        f"Saved frame {saved_count}/{number_to_save} for {camera_name}",
                        end="\r",
                    )

                saved_count += 1
                cooldown = cooldown_time

        cv2.imshow(f"{camera_name} frame", frame_small)

        # If we're using single camera mode, also show the other half
        if use_single_camera and camera_name == "camera0":
            other_frame = cv2.resize(
                right_frame, None, fx=1 / view_resize, fy=1 / view_resize
            )
            cv2.imshow("camera1 (from stereo)", other_frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            print("Frame collection cancelled.")
            break

        if key == 32:  # SPACE key
            start = True

        # Break out of the loop when enough frames have been saved
        if saved_count == number_to_save:
            print(f"Completed collecting {number_to_save} frames for {camera_name}")

            # If we're in single camera mode and this is camera0,
            # we've also collected camera1 frames
            if use_single_camera and camera_name == "camera0":
                print(
                    f"Also collected {number_to_save} frames for camera1 from the same source"
                )

            break

    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera_for_intrinsic_parameters(
    images_prefix: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calibrate a single camera to obtain its intrinsic parameters.

    This function processes images containing checkerboard patterns to calculate
    the camera matrix and distortion coefficients.

    Parameters:
        images_prefix (str): Path prefix for the calibration images (e.g., "frames/camera0*")

    Returns:
        tuple: Camera matrix, distortion coefficients, and RMSE
            - camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
            - distortion_coeffs (numpy.ndarray): Camera distortion coefficients
            - rmse (float): Root Mean Square Error of the calibration
    """
    print(f"Calibrating {images_prefix.split('/')[-1]} intrinsics...")
    # Find all images matching the prefix pattern
    image_paths = sorted(glob.glob(images_prefix))

    if not image_paths:
        print(f"Error: No images found matching {images_prefix}")
        return None, None, float("inf")

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
        return None, None, float("inf")

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None
    )
    print("Camera calibration complete")
    print("RMSE:", ret)
    return camera_matrix, distortion_coeffs, float(ret)


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
    crop_percentage = calibration_settings.get("crop_percentage", 0)

    # Check if using single camera mode for stereo
    use_single_camera = calibration_settings.get("use_single_camera_for_stereo", False)

    if crop_percentage > 0:
        print(f"Cameras will be cropped by {crop_percentage}%")

    # Setup camera capture
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]

    if use_single_camera:
        camera_device_id = calibration_settings[camera0_name]

        # In single camera mode, we only need one camera
        print(f"Using single camera stereo mode with device ID: {camera_device_id}")

        # Open single video stream
        cap = cv2.VideoCapture(camera_device_id)

        # Set resolution to double width for side-by-side stereo
        cap.set(3, width * 2)  # Double width for side-by-side stereo
        cap.set(4, height)

        # Check if camera opened successfully
        if not cap.isOpened():
            print(
                f"Error: Could not open stereo camera (device ID: {camera_device_id})"
            )
            return

    else:
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
        cap0.set(3, width)
        cap0.set(4, height)
        cap1.set(3, width)
        cap1.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        # Get frames based on whether we're using single or dual camera mode
        if use_single_camera:
            ret, frame = cap.read()
            if not ret:
                print("Stereo camera not returning video data. Exiting...")
                break

            # Split the combined frame into left and right
            frame0, frame1 = split_stereo_frame(frame)
        else:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("Cameras not returning video data. Exiting...")
                break

        # Apply fisheye cropping if needed
        if crop_percentage > 0:
            frame0 = crop_frame(frame0, crop_percentage)
            frame1 = crop_frame(frame1, crop_percentage)

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
                0.7,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                frame0_small,
                "Press SPACEBAR to start collection frames",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
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
                    0.7,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Num frames: {saved_count}/{number_to_save}",
                    (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
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

    # Close camera(s)
    if use_single_camera:
        cap.release()
    else:
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
) -> Tuple[np.ndarray, np.ndarray, float]:
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
        Tuple[np.ndarray, np.ndarray, float]:
            - Rotation matrix from camera0 to camera1
            - Translation vector from camera0 to camera1
            - RMSE of the stereo calibration
    """
    print("Performing stereo calibration...")
    # Read the synchronized frame pairs
    c0_images_paths = sorted(glob.glob(frames_prefix_c0))
    c1_images_paths = sorted(glob.glob(frames_prefix_c1))

    if not c0_images_paths or not c1_images_paths:
        print("Error: No stereo image pairs found")
        return None, None, float("inf")

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
        return None, None, float("inf")

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
    return rotation_matrix, translation_vector, float(ret)


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

    cap0 = cv2.VideoCapture(calibration_settings[camera0_name])

    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    crop_percentage = calibration_settings.get("crop_percentage", 0)

    # Check if using single camera mode
    use_single_camera = calibration_settings.get("use_single_camera_for_stereo", False)

    if use_single_camera:
        # Set resolution to double width for side-by-side stereo
        cap0.set(3, width * 2)
        cap0.set(4, height)

        if not cap0.isOpened():
            print(
                f"Error: Could not open stereo camera (device ID: {calibration_settings[camera0_name]})"
            )
            sys.exit(1)

        # Use RGB colors to represent XYZ axes
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z

        while True:
            ret, frame = cap0.read()
            if not ret:
                print("Video stream not returning frame data")
                sys.exit(1)

            # Split the frame into left and right
            frame0, frame1 = split_stereo_frame(frame)
            if crop_percentage > 0:
                frame0 = crop_frame(frame0, crop_percentage)
                frame1 = crop_frame(frame1, crop_percentage)

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
    else:
        # Open video streams
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


def rectify_stereo_fisheye(
    camera_matrix0: np.ndarray,
    dist_coeffs0: np.ndarray,
    camera_matrix1: np.ndarray,
    dist_coeffs1: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Perform stereo rectification for fisheye cameras.

    Parameters:
        camera_matrix0: Intrinsic matrix of the first camera
        dist_coeffs0: Distortion coefficients of the first camera
        camera_matrix1: Intrinsic matrix of the second camera
        dist_coeffs1: Distortion coefficients of the second camera
        rotation_matrix: Rotation matrix from camera0 to camera1
        translation_vector: Translation vector from camera0 to camera1
        image_size: Size of the image (width, height)

    Returns:
        Tuple containing:
        - Dictionary with rectification parameters for camera0
        - Dictionary with rectification parameters for camera1
    """
    print("Performing stereo rectification for fisheye cameras...")

    # Convert distortion coefficients format for fisheye model
    # OpenCV fisheye model expects 4 parameters: k1, k2, k3, k4
    if dist_coeffs0.shape[1] > 4:
        dist_coeffs0 = dist_coeffs0[:, :4]
    if dist_coeffs1.shape[1] > 4:
        dist_coeffs1 = dist_coeffs1[:, :4]

    # Get rectification transforms and projection matrices
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    # Determine the free scaling parameter
    balance = 0.0
    if "fisheye_balance_parameter" in calibration_settings:
        balance = calibration_settings["fisheye_balance_parameter"]

    # Determine the FOV for the rectified images
    fov_scale = 1.0
    if "fisheye_fov_scale" in calibration_settings:
        fov_scale = calibration_settings["fisheye_fov_scale"]

    # Perform stereo rectification
    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(
        camera_matrix0,
        dist_coeffs0,
        camera_matrix1,
        dist_coeffs1,
        image_size,
        rotation_matrix,
        translation_vector,
        flags,
        (0, 0),
        (0, 0),
        None,
        balance=balance,
        fov_scale=fov_scale,
    )

    # Initialize undistortion and rectification maps
    map0_x, map0_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix0, dist_coeffs0, R0, P0, image_size, cv2.CV_32FC1
    )
    map1_x, map1_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix1, dist_coeffs1, R1, P1, image_size, cv2.CV_32FC1
    )

    # Create dictionaries to store all parameters
    camera0_rect = {"R": R0, "P": P0, "map_x": map0_x, "map_y": map0_y}

    camera1_rect = {"R": R1, "P": P1, "map_x": map1_x, "map_y": map1_y}

    # Store disparity-to-depth mapping matrix
    disparity_to_depth = Q

    # Update CalibrationState
    if calibration_state is not None:
        calibration_state.rect_R0 = R0
        calibration_state.rect_P0 = P0
        calibration_state.rect_R1 = R1
        calibration_state.rect_P1 = P1
        calibration_state.disparity_to_depth_map = Q
        calibration_state.save()

    print("Rectification maps created successfully")
    return camera0_rect, camera1_rect


def rectify_stereo_standard(
    camera_matrix0: np.ndarray,
    dist_coeffs0: np.ndarray,
    camera_matrix1: np.ndarray,
    dist_coeffs1: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Perform stereo rectification for standard (non-fisheye) cameras.

    Parameters:
        camera_matrix0: Intrinsic matrix of the first camera
        dist_coeffs0: Distortion coefficients of the first camera
        camera_matrix1: Intrinsic matrix of the second camera
        dist_coeffs1: Distortion coefficients of the second camera
        rotation_matrix: Rotation matrix from camera0 to camera1
        translation_vector: Translation vector from camera0 to camera1
        image_size: Size of the image (width, height)

    Returns:
        Tuple containing:
        - Dictionary with rectification parameters for camera0
        - Dictionary with rectification parameters for camera1
    """
    print("Performing stereo rectification for standard cameras...")

    alpha = calibration_settings.get("rectification_alpha", -1)

    # Perform stereo rectification
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        camera_matrix0,
        dist_coeffs0,
        camera_matrix1,
        dist_coeffs1,
        image_size,
        rotation_matrix,
        translation_vector,
        None,
        None,
        None,
        None,
        None,
        cv2.CALIB_ZERO_DISPARITY,
        alpha,
    )

    # Initialize undistortion and rectification maps
    map0_x, map0_y = cv2.initUndistortRectifyMap(
        camera_matrix0, dist_coeffs0, R0, P0, image_size, cv2.CV_32FC1
    )
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        camera_matrix1, dist_coeffs1, R1, P1, image_size, cv2.CV_32FC1
    )

    # Create dictionaries to store all parameters
    camera0_rect = {"R": R0, "P": P0, "map_x": map0_x, "map_y": map0_y, "roi": roi0}

    camera1_rect = {"R": R1, "P": P1, "map_x": map1_x, "map_y": map1_y, "roi": roi1}

    # Update CalibrationState
    if calibration_state is not None:
        calibration_state.rect_R0 = R0
        calibration_state.rect_P0 = P0
        calibration_state.rect_R1 = R1
        calibration_state.rect_P1 = P1
        calibration_state.disparity_to_depth_map = Q
        calibration_state.roi0 = roi0
        calibration_state.roi1 = roi1
        calibration_state.save()

    print("Rectification maps created successfully")
    print(f"Valid ROI for camera0: {roi0}")
    print(f"Valid ROI for camera1: {roi1}")

    return camera0_rect, camera1_rect


def create_rectification_maps(
    cmtx0: np.ndarray,
    dist0: np.ndarray,
    cmtx1: np.ndarray,
    dist1: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Create rectification maps based on calibration parameters.

    Parameters:
        cmtx0: Intrinsic matrix of camera0
        dist0: Distortion coefficients of camera0
        cmtx1: Intrinsic matrix of camera1
        dist1: Distortion coefficients of camera1
        R: Rotation matrix from camera0 to camera1
        T: Translation vector from camera0 to camera1

    Returns:
        Tuple containing rectification parameters for both cameras
    """
    # Get image dimensions from calibration settings
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]

    # Check for cropping and adjust dimensions if needed
    crop_percentage = calibration_settings.get("crop_percentage", 0)

    # Calculate adjusted dimensions if cropping is applied
    if crop_percentage > 0:
        crop_pixels_h = int(height * crop_percentage / 100)
        crop_pixels_w = int(width * crop_percentage / 100)
        adjusted_width = width - 2 * crop_pixels_w
        adjusted_height = height - 2 * crop_pixels_h
        image_size = (adjusted_width, adjusted_height)
        print(
            f"Using adjusted dimensions for rectification: {image_size} (with {crop_percentage}% crop)"
        )
    else:
        image_size = (width, height)
        print(f"Using original dimensions for rectification: {image_size}")

    # Check camera fisheye setting
    is_fisheye = calibration_settings.get("is_fisheye", False)

    if is_fisheye:  # Both cameras are fisheye
        print("Both cameras are fisheye - using fisheye rectification")
        return rectify_stereo_fisheye(cmtx0, dist0, cmtx1, dist1, R, T, image_size)
    else:  # Both cameras are standard
        print("Both cameras are standard - using standard rectification")
        return rectify_stereo_standard(cmtx0, dist0, cmtx1, dist1, R, T, image_size)


def verify_rectification(
    camera0_rect: Dict[str, np.ndarray],
    camera1_rect: Dict[str, np.ndarray],
    camera0_name: str,
    camera1_name: str,
) -> None:
    """
    Verify rectification by showing rectified video from both cameras.

    Parameters:
        camera0_rect: Rectification parameters for camera0
        camera1_rect: Rectification parameters for camera1
        camera0_name: Name of camera0 in settings
        camera1_name: Name of camera1 in settings
    """
    print("Opening cameras to verify rectification...")

    # Check if using single camera mode
    use_single_camera = calibration_settings.get("use_single_camera_for_stereo", False)

    # Set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    crop_percentage = calibration_settings.get("crop_percentage", 0)

    if use_single_camera:
        # Open single video stream
        cap = cv2.VideoCapture(calibration_settings[camera0_name])

        # Set resolution to double width for side-by-side stereo
        cap.set(3, width * 2)
        cap.set(4, height)

        # Check if camera opened successfully
        if not cap.isOpened():
            print(
                f"Error: Could not open stereo camera (device ID: {calibration_settings[camera0_name]})"
            )
            return
    else:
        # Open video streams
        cap0 = cv2.VideoCapture(calibration_settings[camera0_name])
        cap1 = cv2.VideoCapture(calibration_settings[camera1_name])

        # Set camera resolutions
        cap0.set(3, width)
        cap0.set(4, height)
        cap1.set(3, width)
        cap1.set(4, height)

        # Check if cameras opened successfully
        if not cap0.isOpened() or not cap1.isOpened():
            print("Error: Could not open one or both cameras")
            if cap0.isOpened():
                cap0.release()
            if cap1.isOpened():
                cap1.release()
            return

    # Extract rectification maps
    map0_x = camera0_rect["map_x"]
    map0_y = camera0_rect["map_y"]
    map1_x = camera1_rect["map_x"]
    map1_y = camera1_rect["map_y"]

    # ROI for cropping (may be None for fisheye)
    roi0 = camera0_rect.get("roi", None)
    roi1 = camera1_rect.get("roi", None)

    print("Displaying rectified frames. Press ESC to exit.")

    while True:
        if use_single_camera:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from stereo camera")
                break

            # Split the combined frame into left and right
            frame0, frame1 = split_stereo_frame(frame)
        else:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("Error reading from cameras")
                break

        if crop_percentage > 0:
            frame0 = crop_frame(frame0, crop_percentage)
            frame1 = crop_frame(frame1, crop_percentage)

        # Rectify images
        frame0_rectified = cv2.remap(frame0, map0_x, map0_y, cv2.INTER_LINEAR)
        frame1_rectified = cv2.remap(frame1, map1_x, map1_y, cv2.INTER_LINEAR)

        # Crop to ROI if available
        if roi0 is not None:
            x, y, w, h = roi0
            if w > 0 and h > 0:
                frame0_rectified = frame0_rectified[y : y + h, x : x + w]

        if roi1 is not None:
            x, y, w, h = roi1
            if w > 0 and h > 0:
                frame1_rectified = frame1_rectified[y : y + h, x : x + w]

        # Draw horizontal lines to check rectification
        for i in range(0, height, 50):
            cv2.line(frame0_rectified, (0, i), (width, i), (0, 255, 0), 1)
            cv2.line(frame1_rectified, (0, i), (width, i), (0, 255, 0), 1)

        cv2.imshow("left", frame0_rectified)
        cv2.imshow("right", frame1_rectified)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    # Close camera(s)
    if use_single_camera:
        cap.release()
    else:
        cap0.release()
        cap1.release()

    cv2.destroyAllWindows()


def save_rectification_maps(
    camera0_rect: Dict[str, np.ndarray],
    camera1_rect: Dict[str, np.ndarray],
    directory: str = "camera_parameters",
) -> None:
    """
    Save rectification maps and parameters to files for later use.

    Parameters:
        camera0_rect: Rectification data for camera0
        camera1_rect: Rectification data for camera1
        directory: Directory to save the parameters
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save rectification data using numpy's save function
    np.save(os.path.join(directory, "camera0_mapx.npy"), camera0_rect["map_x"])
    np.save(os.path.join(directory, "camera0_mapy.npy"), camera0_rect["map_y"])
    np.save(os.path.join(directory, "camera0_rect_R.npy"), camera0_rect["R"])
    np.save(os.path.join(directory, "camera0_rect_P.npy"), camera0_rect["P"])

    np.save(os.path.join(directory, "camera1_mapx.npy"), camera1_rect["map_x"])
    np.save(os.path.join(directory, "camera1_mapy.npy"), camera1_rect["map_y"])
    np.save(os.path.join(directory, "camera1_rect_R.npy"), camera1_rect["R"])
    np.save(os.path.join(directory, "camera1_rect_P.npy"), camera1_rect["P"])

    # Save ROI if available
    if "roi" in camera0_rect:
        np.save(os.path.join(directory, "camera0_roi.npy"), camera0_rect["roi"])
    if "roi" in camera1_rect:
        np.save(os.path.join(directory, "camera1_roi.npy"), camera1_rect["roi"])

    print(f"Rectification maps and parameters saved to {directory}")


def modify_calibration_state_for_rectification():
    """
    Update the CalibrationState class to include rectification parameters.
    This function needs to be called before using the CalibrationState class.
    """
    # Extend CalibrationState with rectification parameters
    CalibrationState.rect_R0 = None
    CalibrationState.rect_P0 = None
    CalibrationState.rect_R1 = None
    CalibrationState.rect_P1 = None
    CalibrationState.disparity_to_depth_map = None
    CalibrationState.roi0 = None
    CalibrationState.roi1 = None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Call with settings filename: "python3 calibrate.py calibration_settings.yaml"'
        )
        sys.exit(1)

    # Open and parse the settings file
    calibration_settings = parse_calibration_settings_file(sys.argv[1])

    # Check if we're using single camera mode and update settings accordingly
    use_single_camera = calibration_settings.get("use_single_camera_for_stereo", False)
    if use_single_camera:
        print("Using single camera for stereo mode - will split frames horizontally")

    # Extend CalibrationState for rectification parameters
    modify_calibration_state_for_rectification()

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

    # Camera 1 - potentially skip in single camera mode since we already collected both
    if use_single_camera:
        # In single camera mode, both camera frames should be collected together
        if not calibration_state.frames_collected_cam1:
            calibration_state.frames_collected_cam1 = True
            calibration_state.stereo_frames_collected = True
            calibration_state.save()
            print(
                "Camera 1 frames collected during camera0 collection (single camera mode)"
            )
    else:
        # Regular dual camera mode
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
        cmtx0, dist0, rmse0 = calibrate_camera_for_intrinsic_parameters(images_prefix)

        if cmtx0 is not None and dist0 is not None:
            calibration_state.camera_matrix0 = cmtx0
            calibration_state.dist_coeffs0 = dist0
            calibration_state.rmse_camera0 = rmse0
            calibration_state.intrinsics_calibrated_cam0 = True
            calibration_state.save()
    else:
        print("Camera 0 intrinsics already calibrated. Loading parameters...")
        cmtx0 = calibration_state.camera_matrix0
        dist0 = calibration_state.dist_coeffs0
        rmse0 = calibration_state.rmse_camera0
        print(f"Camera 0 calibration RMSE: {rmse0:.6f}")

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to recalibrate camera0 intrinsics?"):
            images_prefix = os.path.join("frames", "camera0*")
            cmtx0, dist0, rmse0 = calibrate_camera_for_intrinsic_parameters(
                images_prefix
            )

            if cmtx0 is not None and dist0 is not None:
                calibration_state.camera_matrix0 = cmtx0
                calibration_state.dist_coeffs0 = dist0
                calibration_state.rmse_camera0 = rmse0
                calibration_state.intrinsics_calibrated_cam0 = True
                calibration_state.save()

    # Camera 1 intrinsics
    if not calibration_state.intrinsics_calibrated_cam1:
        images_prefix = os.path.join("frames", "camera1*")
        cmtx1, dist1, rmse1 = calibrate_camera_for_intrinsic_parameters(images_prefix)

        if cmtx1 is not None and dist1 is not None:
            calibration_state.camera_matrix1 = cmtx1
            calibration_state.dist_coeffs1 = dist1
            calibration_state.rmse_camera1 = rmse1
            calibration_state.intrinsics_calibrated_cam1 = True
            calibration_state.save()
    else:
        print("Camera 1 intrinsics already calibrated. Loading parameters...")
        cmtx1 = calibration_state.camera_matrix1
        dist1 = calibration_state.dist_coeffs1
        rmse1 = calibration_state.rmse_camera1
        print(f"Camera 1 calibration RMSE: {rmse1:.6f}")

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to recalibrate camera1 intrinsics?"):
            images_prefix = os.path.join("frames", "camera1*")
            cmtx1, dist1, rmse1 = calibrate_camera_for_intrinsic_parameters(
                images_prefix
            )

            if cmtx1 is not None and dist1 is not None:
                calibration_state.camera_matrix1 = cmtx1
                calibration_state.dist_coeffs1 = dist1
                calibration_state.rmse_camera1 = rmse1
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

        R1, T1, rmse_stereo = stereo_calibrate(
            cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
        )

        if R1 is not None and T1 is not None:
            calibration_state.rotation_matrix0 = R0
            calibration_state.translation_vector0 = T0
            calibration_state.rotation_matrix1 = R1
            calibration_state.translation_vector1 = T1
            calibration_state.rmse_stereo = rmse_stereo
            calibration_state.stereo_calibrated = True
            calibration_state.save()
    else:
        print("Stereo calibration already completed. Loading parameters...")
        R0 = calibration_state.rotation_matrix0
        T0 = calibration_state.translation_vector0
        R1 = calibration_state.rotation_matrix1
        T1 = calibration_state.translation_vector1
        rmse_stereo = calibration_state.rmse_stereo
        print(f"Stereo calibration RMSE: {rmse_stereo:.6f}")

        # Option to recalibrate if needed
        if ask_yes_no_question("Do you want to perform stereo calibration again?"):
            frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
            frames_prefix_c1 = os.path.join("frames_pair", "camera1*")
            R1, T1, rmse_stereo = stereo_calibrate(
                cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1
            )

            if R1 is not None and T1 is not None:
                calibration_state.rotation_matrix1 = R1
                calibration_state.translation_vector1 = T1
                calibration_state.rmse_stereo = rmse_stereo
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

    """Step6. Generate rectification maps for stereo vision."""
    print_step_banner(6, "GENERATE RECTIFICATION MAPS")

    # Check if rectification is needed
    should_rectify = calibration_settings.get("perform_rectification", False)

    if should_rectify and calibration_state.stereo_calibrated:
        print("Generating rectification maps...")

        # Create rectification maps
        camera0_rect, camera1_rect = create_rectification_maps(
            cmtx0, dist0, cmtx1, dist1, R1, T1
        )

        # Save rectification maps for later use
        save_rectification_maps(camera0_rect, camera1_rect)

        verify_rectification(camera0_rect, camera1_rect, "camera0", "camera1")
    elif should_rectify:
        print("Cannot generate rectification maps: stereo calibration not completed")
    else:
        print("Rectification step skipped (not enabled in settings)")

    """Optional. Define a different origin point and save the calibration data"""
    print_step_banner(7, "OPTIONAL: DEFINE ALTERNATIVE WORLD ORIGIN (OPTIONAL)")

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
