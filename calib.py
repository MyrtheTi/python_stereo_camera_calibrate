"""
Stereo Camera Calibration

This script provides functionality for calibrating stereo camera systems.
It handles intrinsic and extrinsic parameter calibration for stereo vision
applications using checkerboard pattern detection.
"""

import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    CalibrationState,
    crop_frame,
    parse_calibration_settings_file,
    split_stereo_frame,
)


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


class StereoCalibrator:
    """
    Class to handle stereo camera calibration.

    This class provides methods to collect frames, calibrate cameras, and
    perform stereo calibration. It also includes methods for saving and
    loading calibration state.
    """

    def __init__(self, settings_file: str) -> None:
        """
        Initialize the StereoCalibrator with settings from a file.

        Parameters:
            settings_file: Path to the calibration settings YAML file
        """
        self.calibration_settings = parse_calibration_settings_file(settings_file)

        # Extract common settings
        self.use_single_camera = self.calibration_settings.get(
            "use_single_camera_for_stereo", False
        )
        self.width = self.calibration_settings["frame_width"]
        self.height = self.calibration_settings["frame_height"]
        self.camera0 = self.calibration_settings["camera0"]
        self.camera1 = self.calibration_settings["camera1"]
        self.number_to_save = self.calibration_settings["calibration_frames"]
        self.view_resize = self.calibration_settings["view_resize"]
        self.cooldown_time = self.calibration_settings["cooldown"]
        self.crop_percentage = self.calibration_settings.get("crop_percentage", 0)

        # Checkerboard pattern settings
        self.rows = self.calibration_settings["checkerboard_rows"]
        self.columns = self.calibration_settings["checkerboard_columns"]
        self.world_scaling = self.calibration_settings["checkerboard_box_size_scale"]

        # Calibration state and working directories
        self.calibration_state = CalibrationState.load()
        self.create_folders()

    def create_folders(self) -> None:
        """Create necessary directories for calibration data."""
        folders = ["frames", "frames_pair", "camera_parameters"]
        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)
                print(f"Created directory: {folder}")

    def open_camera(self, camera_id: int, is_stereo: bool = False) -> cv2.VideoCapture:
        """
        Open and configure a camera.

        Parameters:
            camera_id: Camera device ID
            is_stereo: Whether this is a stereo camera requiring double-width

        Returns:
            Configured VideoCapture object
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera (device ID: {camera_id})")

        # Configure resolution
        if is_stereo:
            cap.set(3, self.width * 2)  # Double width for side-by-side stereo
        else:
            cap.set(3, self.width)
        cap.set(4, self.height)

        return cap

    def get_frames_from_cameras(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frames from cameras based on configuration.

        Returns:
            Tuple containing left and right frames
        """
        if self.use_single_camera:
            cap = self.open_camera(self.camera0, is_stereo=True)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Failed to get frame from stereo camera")

            # Split stereo frame
            frame0, frame1 = split_stereo_frame(frame)
        else:
            cap0 = self.open_camera(self.camera0)
            cap1 = self.open_camera(self.camera1)

            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            cap0.release()
            cap1.release()

            if not ret0 or not ret1:
                raise RuntimeError("Failed to get frames from cameras")

        # Apply cropping if needed
        if self.crop_percentage > 0:
            frame0 = crop_frame(frame0, self.crop_percentage)
            frame1 = crop_frame(frame1, self.crop_percentage)

        return frame0, frame1

    # --- UI Helpers ---

    def put_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        colour: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        """
        Add text to a frame.

        Parameters:
            frame: The image frame
            text: Text to display
            position: (x, y) coordinates for text
            colour: BGR colour tuple
        """
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            colour,
            1,
        )

    def resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a frame for display purposes.

        Parameters:
            frame: Original frame

        Returns:
            Resized frame
        """
        if self.view_resize == 1:
            return frame
        return cv2.resize(frame, None, fx=1 / self.view_resize, fy=1 / self.view_resize)

    def detect_checkerboard(
        self, frame: np.ndarray, refine_corners: bool = True
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard pattern in a frame.

        Parameters:
            frame: Input image
            refine_corners: Whether to perform subpixel corner refinement

        Returns:
            Tuple containing:
            - Success flag
            - Corner points (if found, otherwise None)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard
        found, corners = cv2.findChessboardCorners(
            gray, (self.rows, self.columns), None
        )

        if found and refine_corners:
            # Criteria for corner refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return found, corners if found else None

    def display_checkerboard(
        self, display_frame: np.ndarray, corners: np.ndarray, mark_origin: bool = False
    ) -> np.ndarray:
        """
        Draw detected checkerboard on a frame.

        Parameters:
            frame: Image frame
            corners: Detected corner points
            mark_origin: Whether to mark the first corner with "O"

        Returns:
            Frame with visualized checkerboard
        """
        # Draw the corners
        cv2.drawChessboardCorners(
            display_frame, (self.rows, self.columns), corners, True
        )

        # Mark the first corner as origin
        if mark_origin:
            origin_point = corners[0, 0].astype(np.int32)
            self.put_text(
                display_frame,
                "O",
                (origin_point[0], origin_point[1]),
            )

        return display_frame

    def save_frames_single_camera(self, camera_name: str) -> None:
        """
        Collect and save frames for single camera calibration.

        Parameters:
            camera_name: Name of camera ("camera0" or "camera1")
        """
        print(f"Starting frame collection for {camera_name}...")

        # Skip camera1 in single-camera mode as it's handled when collecting camera0
        if self.use_single_camera and camera_name == "camera1":
            print(
                "Using single camera for stereo - camera1 frames will be captured with camera0"
            )
            return

        if self.crop_percentage > 0:
            print(f"Camera {camera_name} will be cropped by {self.crop_percentage}%")

        # Get camera device ID
        camera_device_id = self.camera0 if camera_name == "camera0" else self.camera1

        if self.use_single_camera:
            print(f"Using single camera stereo mode with device ID: {camera_device_id}")

        # Open camera
        try:
            cap = self.open_camera(camera_device_id, is_stereo=self.use_single_camera)
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            return

        cooldown = self.cooldown_time
        start = False
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No video data received from camera. Exiting...")
                break

            # Process frames based on camera mode
            if self.use_single_camera:
                left_frame, right_frame = split_stereo_frame(frame)
                frame = left_frame if camera_name == "camera0" else right_frame

            # Apply cropping if needed
            if self.crop_percentage > 0:
                frame = crop_frame(frame, self.crop_percentage)
                if self.use_single_camera:
                    right_frame = crop_frame(right_frame, self.crop_percentage)

            # Resize for display
            display_frame = frame.copy()
            display_frame = self.resize_for_display(display_frame)

            # Show instructions or progress
            if not start:
                self.put_text(
                    display_frame, "Press SPACEBAR to start collection frames", (50, 50)
                )
            else:
                cooldown -= 1
                self.put_text(display_frame, f"Cooldown: {cooldown}", (50, 50))
                self.put_text(
                    display_frame,
                    f"Num frames: {saved_count}/{self.number_to_save}",
                    (50, 100),
                )

                # Save the frame when cooldown reaches 0
                if cooldown <= 0:
                    self._save_calibration_frame(camera_name, saved_count, frame)

                    # For single camera mode, also save the other view
                    if self.use_single_camera and camera_name == "camera0":
                        self._save_calibration_frame(
                            "camera1", saved_count, right_frame
                        )

                        # Also save to stereo pairs folder
                        self._save_stereo_pair(saved_count, frame, right_frame)

                        print(
                            f"Saved both camera0 and camera1 frames {saved_count+1}/{self.number_to_save}",
                            end="\r",
                        )
                    else:
                        print(
                            f"Saved frame {saved_count+1}/{self.number_to_save} for {camera_name}",
                            end="\r",
                        )

                    saved_count += 1
                    cooldown = self.cooldown_time

            # Display frames
            cv2.imshow(f"{camera_name} frame", display_frame)

            # For single camera mode, also show the other half
            if self.use_single_camera and camera_name == "camera0":
                other_frame = self.resize_for_display(right_frame)
                cv2.imshow("camera1 (from stereo)", other_frame)

            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("\nFrame collection cancelled.")
                break
            if key == 32:  # SPACE key
                start = True

            # Break when enough frames are collected
            if saved_count == self.number_to_save:
                print(
                    f"\nCompleted collecting {self.number_to_save} frames for {camera_name}"
                )

                if self.use_single_camera and camera_name == "camera0":
                    print(
                        f"Also collected {self.number_to_save} frames for camera1 from the same source"
                    )
                break

        cap.release()
        cv2.destroyAllWindows()

    def _save_calibration_frame(
        self, camera_name: str, frame_index: int, frame: np.ndarray
    ) -> None:
        """Save a calibration frame to disk."""
        path = os.path.join("frames", f"{camera_name}_{frame_index}.png")
        cv2.imwrite(path, frame)

    def _save_stereo_pair(
        self, frame_index: int, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> None:
        """Save a stereo frame pair to disk."""
        cv2.imwrite(
            os.path.join("frames_pair", f"camera0_{frame_index}.png"), left_frame
        )
        cv2.imwrite(
            os.path.join("frames_pair", f"camera1_{frame_index}.png"), right_frame
        )

    def save_frames_two_cams(self, camera0_name: str, camera1_name: str) -> None:
        """
        Collect and save synchronized stereo frame pairs.

        Parameters:
            camera0_name: Name of first camera ("camera0")
            camera1_name: Name of second camera ("camera1")
        """
        print(
            f"Starting synchronized frame collection for {camera0_name} and {camera1_name}..."
        )

        if self.crop_percentage > 0:
            print(f"Cameras will be cropped by {self.crop_percentage}%")

        # Open camera(s) based on mode
        try:
            if self.use_single_camera:
                cap = self.open_camera(self.camera0, is_stereo=True)
            else:
                cap0 = self.open_camera(self.camera0)
                cap1 = self.open_camera(self.camera1)
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            return

        cooldown = self.cooldown_time
        start = False
        saved_count = 0

        try:
            while True:
                # Get frames from camera(s)
                if self.use_single_camera:
                    ret, frame = cap.read()
                    if not ret:
                        print("Stereo camera not returning video data. Exiting...")
                        break
                    frame0, frame1 = split_stereo_frame(frame)
                else:
                    ret0, frame0 = cap0.read()
                    ret1, frame1 = cap1.read()
                    if not ret0 or not ret1:
                        print("Cameras not returning video data. Exiting...")
                        break

                # Apply cropping
                if self.crop_percentage > 0:
                    frame0 = crop_frame(frame0, self.crop_percentage)
                    frame1 = crop_frame(frame1, self.crop_percentage)

                # Create smaller versions for display
                display_frame0 = frame0.copy()
                display_frame1 = frame1.copy()
                display_frame0 = self.resize_for_display(display_frame0)
                display_frame1 = self.resize_for_display(display_frame1)

                # Show instructions or progress
                if not start:
                    self.put_text(
                        display_frame0,
                        "Make sure both cameras can see the calibration pattern well",
                        (50, 50),
                    )
                    self.put_text(
                        display_frame0,
                        "Press SPACEBAR to start collection frames",
                        (50, 100),
                    )
                else:
                    cooldown -= 1
                    # Update displays with cooldown and frame count
                    for frame in [display_frame0, display_frame1]:
                        self.put_text(
                            frame, f"Cooldown: {cooldown}", (50, 50), (0, 255, 0)
                        )
                        self.put_text(
                            frame,
                            f"Num frames: {saved_count}/{self.number_to_save}",
                            (50, 100),
                            (0, 255, 0),
                        )

                    # Save frame pair when cooldown reaches zero
                    if cooldown <= 0:
                        self._save_stereo_pair(saved_count, frame0, frame1)
                        saved_count += 1
                        print(
                            f"Saved stereo frame pair {saved_count}/{self.number_to_save}",
                            end="\r",
                        )
                        cooldown = self.cooldown_time

                # Display frames
                cv2.imshow("Camera 0", display_frame0)
                cv2.imshow("Camera 1", display_frame1)

                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("\nFrame collection cancelled.")
                    break
                if key == 32:  # SPACE key
                    start = True
                    print("Stereo frame collection started...")

                # Exit when enough frames are collected
                if saved_count == self.number_to_save:
                    print(
                        f"\nCompleted collecting {self.number_to_save} stereo frame pairs"
                    )
                    break
        finally:
            # Release camera resources
            if self.use_single_camera:
                cap.release()
            else:
                cap0.release()
                cap1.release()
            cv2.destroyAllWindows()

    def get_checkerboard_object_points(self) -> np.ndarray:
        """
        Generate 3D coordinates for checkerboard corners.

        Returns:
            np.ndarray: 3D coordinates of checkerboard points
        """
        # Create the standard checkerboard object points
        points = np.zeros((self.rows * self.columns, 3), np.float32)
        points[:, :2] = np.mgrid[0 : self.rows, 0 : self.columns].T.reshape(-1, 2)
        return self.world_scaling * points

    def calibrate_camera_for_intrinsic_parameters(
        self,
        images_prefix: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Calibrate single camera intrinsic parameters.

        Parameters:
            images_prefix: Path pattern for calibration images

        Returns:
            Tuple containing:
            - Camera matrix
            - Distortion coefficients
            - RMSE of calibration
        """
        print(f"Calibrating {images_prefix.split('/')[-1]} intrinsics...")

        # Find all images
        image_paths = sorted(glob.glob(images_prefix))
        if not image_paths:
            print(f"Error: No images found matching {images_prefix}")
            return None, None, float("inf")

        print(f"Found {len(image_paths)} calibration images")

        # Read all calibration frames
        images = [cv2.imread(image_path, 1) for image_path in image_paths]

        # Get frame dimensions
        height, width = images[0].shape[:2]

        # Create object points template
        checkerboard_points_3d = self.get_checkerboard_object_points()

        # Store detected points
        object_points = []  # 3D points in real world space
        image_points = []  # 2D points in image plane

        # Process each frame
        total_frames = len(images)
        for i, frame in enumerate(images):
            print(f"Processing frame {i+1}/{total_frames}", end="\r")

            # Detect checkerboard
            found, corners = self.detect_checkerboard(frame)

            # Prepare display frame
            display_frame = frame.copy()
            self.put_text(display_frame, f"Frame {i+1}/{total_frames}", (25, 25))

            if found:
                # Draw the checkerboard
                cv2.drawChessboardCorners(
                    display_frame, (self.rows, self.columns), corners, found
                )
                self.put_text(
                    display_frame,
                    'Press "s" to skip, any other key to use this image',
                    (25, 50),
                )

                # Show the frame
                cv2.imshow("Calibration Image", display_frame)
                k = cv2.waitKey(0)

                # Skip if requested
                if k & 0xFF == ord("s"):
                    print(f"\nSkipping frame {i+1}")
                    continue

                # Store points
                object_points.append(checkerboard_points_3d)
                image_points.append(corners)
            else:
                print(f"No checkerboard found in frame {i+1}")
                self.put_text(
                    display_frame,
                    "No checkerboard detected! Press any key to continue.",
                    (25, 50),
                )
                cv2.imshow("Calibration Image", display_frame)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        print("\nPerforming camera calibration...")

        # Check if we have enough points
        if not object_points:
            print("Error: No valid calibration frames detected")
            return None, None, float("inf")

        # Perform calibration
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (width, height), None, None
        )

        print("Camera calibration complete")
        print(f"RMSE: {ret:.6f}")

        return camera_matrix, distortion_coeffs, float(ret)

    def stereo_calibrate(
        self,
        mtx0: np.ndarray,
        dist0: np.ndarray,
        mtx1: np.ndarray,
        dist1: np.ndarray,
        frames_prefix_c0: str,
        frames_prefix_c1: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Calculate stereo camera calibration parameters.

        Parameters:
            mtx0: Camera matrix for camera0
            dist0: Distortion coefficients for camera0
            mtx1: Camera matrix for camera1
            dist1: Distortion coefficients for camera1
            frames_prefix_c0: Path pattern for camera0 stereo frames
            frames_prefix_c1: Path pattern for camera1 stereo frames

        Returns:
            Tuple containing:
            - Rotation matrix from camera0 to camera1
            - Translation vector from camera0 to camera1
            - RMSE of stereo calibration
        """
        print("Performing stereo calibration...")

        # Find stereo image pairs
        c0_images_paths = sorted(glob.glob(frames_prefix_c0))
        c1_images_paths = sorted(glob.glob(frames_prefix_c1))

        if not c0_images_paths or not c1_images_paths:
            print("Error: No stereo image pairs found")
            return None, None, float("inf")

        if len(c0_images_paths) != len(c1_images_paths):
            print(
                f"Warning: Number of frames doesn't match between cameras: "
                f"{len(c0_images_paths)} vs {len(c1_images_paths)}"
            )

        # Load image pairs
        total_pairs = min(len(c0_images_paths), len(c1_images_paths))
        print(f"Found {total_pairs} stereo image pairs")

        c0_images = [cv2.imread(path, 1) for path in c0_images_paths[:total_pairs]]
        c1_images = [cv2.imread(path, 1) for path in c1_images_paths[:total_pairs]]

        # Create object points template
        checkerboard_points_3d = self.get_checkerboard_object_points()

        # Get frame dimensions
        height, width = c0_images[0].shape[:2]

        # Store detected points
        image_points_left = []  # 2D points in camera0 image plane
        image_points_right = []  # 2D points in camera1 image plane
        object_points = []  # 3D points in checkerboard space

        # Process each frame pair
        used_pairs = 0
        for i, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
            print(f"Processing stereo pair {i+1}/{total_pairs}", end="\r")

            # Detect checkerboards in both images
            found0, corners0 = self.detect_checkerboard(frame0)
            found1, corners1 = self.detect_checkerboard(frame1)

            display_frame0 = frame0.copy()
            display_frame1 = frame1.copy()

            # If checkerboard is found in both images
            if found0 and found1:
                # Mark the first corner as origin and draw checkerboard
                display_frame0 = self.display_checkerboard(
                    display_frame0, corners0, mark_origin=True
                )
                display_frame1 = self.display_checkerboard(
                    display_frame1, corners1, mark_origin=True
                )

                # Add progress indicators
                for frame in [display_frame0, display_frame1]:
                    self.put_text(
                        frame,
                        f"Pair {i+1}/{total_pairs} - Used: {used_pairs}",
                        (25, 25),
                    )
                    self.put_text(
                        frame,
                        'Press "s" to skip, any other key to use this pair',
                        (25, 50),
                    )

                # Display frames with detected corners
                cv2.imshow("Camera 0", display_frame0)
                cv2.imshow("Camera 1", display_frame1)
                key = cv2.waitKey(0)

                # Skip if requested
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
                if not found0:
                    self.put_text(display_frame0, "No checkerboard detected!", (25, 50))
                if not found1:
                    self.put_text(display_frame1, "No checkerboard detected!", (25, 50))

                cv2.imshow("Camera 0", display_frame0)
                cv2.imshow("Camera 1", display_frame1)
                cv2.waitKey(500)  # Show briefly and continue

        cv2.destroyAllWindows()
        print(f"Using {used_pairs}/{total_pairs} stereo pairs for calibration")

        # Check if we have enough pairs
        if used_pairs == 0:
            print("Error: No valid stereo pairs found. Cannot perform calibration.")
            return None, None, float("inf")

        # Calibration criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Perform stereo calibration
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
        print(f"Stereo calibration RMSE: {ret:.6f}")

        return rotation_matrix, translation_vector, float(ret)

    def make_homogeneous_transform(
        self, rotation_matrix: np.ndarray, translation_vector: np.ndarray
    ) -> np.ndarray:
        """
        Create a 4x4 homogeneous transformation matrix.

        Parameters:
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector

        Returns:
            4x4 homogeneous transformation matrix
        """
        transform_matrix = np.zeros((4, 4))
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation_vector.reshape(3)
        transform_matrix[3, 3] = 1
        return transform_matrix

    def get_projection_matrix(
        self,
        camera_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a camera projection matrix.

        Parameters:
            camera_matrix: 3x3 intrinsic matrix
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector

        Returns:
            3x4 projection matrix
        """
        homogeneous_matrix = self.make_homogeneous_transform(
            rotation_matrix, translation_vector
        )
        projection_matrix = camera_matrix @ homogeneous_matrix[:3, :]
        return projection_matrix

    def check_calibration(
        self,
        camera0_name: str,
        camera0_data: List[np.ndarray],
        camera1_name: str,
        camera1_data: List[np.ndarray],
        _zshift: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verify calibration by projecting 3D axes onto camera views.

        Parameters:
            camera0_name: Name of first camera
            camera0_data: List of [camera_matrix, dist_coeffs, R, T] for camera0
            camera1_name: Name of second camera
            camera1_data: List of [camera_matrix, dist_coeffs, R, T] for camera1
            _zshift: Z-distance for better visibility of axes

        Returns:
            Tuple of projection matrices for both cameras
        """
        print("Verifying calibration results...")

        # Extract camera parameters
        cmtx0, dist0, R0, T0 = [np.array(x) for x in camera0_data]
        cmtx1, dist1, R1, T1 = [np.array(x) for x in camera1_data]

        # Calculate projection matrices
        P0 = self.get_projection_matrix(cmtx0, R0, T0)
        P1 = self.get_projection_matrix(cmtx1, R1, T1)

        # Create 3D coordinate system with origin and axes
        coordinate_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Origin
                [1.0, 0.0, 0.0],  # X axis
                [0.0, 1.0, 0.0],  # Y axis
                [0.0, 0.0, 1.0],  # Z axis
            ]
        )

        # Shift axes in Z direction for better visibility
        axes_points_3d = 5 * coordinate_points + np.array([0.0, 0.0, _zshift]).reshape(
            (1, 3)
        )

        # Project points to each camera view
        pixel_points = {}
        for camera_idx, P in enumerate([P0, P1]):
            points = []
            for point_3d in axes_points_3d:
                # Convert to homogeneous coordinates
                point_homogeneous = np.append(point_3d, 1.0)

                # Project to camera view
                point_projected = P @ point_homogeneous
                pixel_coords = (
                    np.array([point_projected[0], point_projected[1]])
                    / point_projected[2]
                )
                points.append(pixel_coords)
            pixel_points[camera_idx] = np.array(points)

        # Display calibration axes on camera views
        self._show_calibration_axes(pixel_points[0], pixel_points[1])

        return P0, P1

    def _show_calibration_axes(
        self, pixel_points0: np.ndarray, pixel_points1: np.ndarray
    ) -> None:
        """
        Show calibration axes on camera views.

        Parameters:
            pixel_points0: Projected points for camera0
            pixel_points1: Projected points for camera1
        """
        # RGB colours for XYZ axes
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        try:
            # Open camera(s)
            if self.use_single_camera:
                cap = self.open_camera(self.camera0, is_stereo=True)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Video stream not returning frame data")
                        break

                    # Split and process stereo frame
                    frame0, frame1 = split_stereo_frame(frame)

                    if self.crop_percentage > 0:
                        frame0 = crop_frame(frame0, self.crop_percentage)
                        frame1 = crop_frame(frame1, self.crop_percentage)

                    # Draw axes on both views
                    self._draw_axes(frame0, pixel_points0, colours)
                    self._draw_axes(frame1, pixel_points1, colours)

                    # Display frames
                    cv2.imshow("Camera 0", frame0)
                    cv2.imshow("Camera 1", frame1)

                    if cv2.waitKey(1) == 27:  # ESC to exit
                        break

                cap.release()
            else:
                cap0 = self.open_camera(self.camera0)
                cap1 = self.open_camera(self.camera1)

                while True:
                    ret0, frame0 = cap0.read()
                    ret1, frame1 = cap1.read()

                    if not ret0 or not ret1:
                        print("Video stream not returning frame data")
                        break

                    if self.crop_percentage > 0:
                        frame0 = crop_frame(frame0, self.crop_percentage)
                        frame1 = crop_frame(frame1, self.crop_percentage)

                    # Draw axes on both views
                    self._draw_axes(frame0, pixel_points0, colours)
                    self._draw_axes(frame1, pixel_points1, colours)

                    # Display frames
                    cv2.imshow("Camera 0", frame0)
                    cv2.imshow("Camera 1", frame1)

                    if cv2.waitKey(1) == 27:  # ESC to exit
                        break

                cap0.release()
                cap1.release()
        except Exception as e:
            print(f"Error during calibration verification: {e}")
        finally:
            cv2.destroyAllWindows()

    def _draw_axes(
        self, frame: np.ndarray, points: np.ndarray, colours: List[Tuple[int, int, int]]
    ) -> None:
        """
        Draw coordinate axes on a frame.

        Parameters:
            frame: Image frame
            points: Projected points (origin + 3 axes)
            colours: Colours for the 3 axes
        """
        # Get origin point
        origin = tuple(points[0].astype(np.int32))

        # Draw each axis line
        for colour, point in zip(colours, points[1:]):
            point_pixel = tuple(point.astype(np.int32))
            cv2.line(frame, origin, point_pixel, colour, 2)

    def rectify_stereo_fisheye(
        self,
        camera_matrix0: np.ndarray,
        dist_coeffs0: np.ndarray,
        camera_matrix1: np.ndarray,
        dist_coeffs1: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Rectify a stereo fisheye camera pair.

        Parameters:
            camera_matrix0: Camera matrix for camera0
            dist_coeffs0: Distortion coefficients for camera0
            camera_matrix1: Camera matrix for camera1
            dist_coeffs1: Distortion coefficients for camera1
            rotation_matrix: Rotation matrix between cameras
            translation_vector: Translation vector between cameras
            image_size: Size (width, height) of rectified images

        Returns:
            Tuple of rectification parameters for both cameras
        """
        print("Performing stereo rectification for fisheye cameras...")

        # Ensure correct distortion coefficient format (4 parameters for fisheye)
        if dist_coeffs0.shape[1] > 4:
            dist_coeffs0 = dist_coeffs0[:, :4]
        if dist_coeffs1.shape[1] > 4:
            dist_coeffs1 = dist_coeffs1[:, :4]

        # Set flags for stereo rectification
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW
        )

        # Get parameters from settings
        balance = self.calibration_settings.get("fisheye_balance_parameter", 0.0)
        fov_scale = self.calibration_settings.get("fisheye_fov_scale", 1.0)

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

        # Create rectification maps
        map0_x, map0_y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix0, dist_coeffs0, R0, P0, image_size, cv2.CV_32FC1
        )
        map1_x, map1_y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix1, dist_coeffs1, R1, P1, image_size, cv2.CV_32FC1
        )

        # Create result dictionaries
        camera0_rect = {"R": R0, "P": P0, "map_x": map0_x, "map_y": map0_y}
        camera1_rect = {"R": R1, "P": P1, "map_x": map1_x, "map_y": map1_y}

        # Update calibration state
        self._update_calibration_state_rectification(R0, P0, R1, P1, Q)

        print("Fisheye rectification maps created successfully")
        return camera0_rect, camera1_rect

    def rectify_stereo_standard(
        self,
        camera_matrix0: np.ndarray,
        dist_coeffs0: np.ndarray,
        camera_matrix1: np.ndarray,
        dist_coeffs1: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Rectify a standard (non-fisheye) stereo camera pair.

        Parameters:
            camera_matrix0: Camera matrix for camera0
            dist_coeffs0: Distortion coefficients for camera0
            camera_matrix1: Camera matrix for camera1
            dist_coeffs1: Distortion coefficients for camera1
            rotation_matrix: Rotation matrix between cameras
            translation_vector: Translation vector between cameras
            image_size: Size (width, height) of rectified images

        Returns:
            Tuple of rectification parameters for both cameras
        """
        print("Performing stereo rectification for standard cameras...")

        # Get alpha parameter from settings
        alpha = self.calibration_settings.get("rectification_alpha", -1)

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

        # Create rectification maps
        map0_x, map0_y = cv2.initUndistortRectifyMap(
            camera_matrix0, dist_coeffs0, R0, P0, image_size, cv2.CV_32FC1
        )
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            camera_matrix1, dist_coeffs1, R1, P1, image_size, cv2.CV_32FC1
        )

        # Create result dictionaries
        camera0_rect = {"R": R0, "P": P0, "map_x": map0_x, "map_y": map0_y, "roi": roi0}
        camera1_rect = {"R": R1, "P": P1, "map_x": map1_x, "map_y": map1_y, "roi": roi1}

        # Update calibration state
        self._update_calibration_state_rectification(R0, P0, R1, P1, Q, roi0, roi1)

        print("Standard rectification maps created successfully")
        print(f"Valid ROI for camera0: {roi0}")
        print(f"Valid ROI for camera1: {roi1}")

        return camera0_rect, camera1_rect

    def _update_calibration_state_rectification(
        self,
        R0: np.ndarray,
        P0: np.ndarray,
        R1: np.ndarray,
        P1: np.ndarray,
        Q: np.ndarray,
        roi0: Optional[Tuple[int, int, int, int]] = None,
        roi1: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Update calibration state with rectification parameters."""
        if self.calibration_state is not None:
            self.calibration_state.rect_R0 = R0
            self.calibration_state.rect_P0 = P0
            self.calibration_state.rect_R1 = R1
            self.calibration_state.rect_P1 = P1
            self.calibration_state.disparity_to_depth_map = Q

            if roi0 is not None:
                self.calibration_state.roi0 = roi0
            if roi1 is not None:
                self.calibration_state.roi1 = roi1

            self.calibration_state.save()

    def create_rectification_maps(
        self,
        cmtx0: np.ndarray,
        dist0: np.ndarray,
        cmtx1: np.ndarray,
        dist1: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Create rectification maps for stereo vision.

        Parameters:
            cmtx0: Camera matrix for camera0
            dist0: Distortion coefficients for camera0
            cmtx1: Camera matrix for camera1
            dist1: Distortion coefficients for camera1
            R: Rotation matrix between cameras
            T: Translation vector between cameras

        Returns:
            Tuple of rectification parameters for both cameras
        """
        # Calculate image dimensions after cropping
        if self.crop_percentage > 0:
            crop_pixels_h = int(self.height * self.crop_percentage / 100)
            crop_pixels_w = int(self.width * self.crop_percentage / 100)
            adjusted_width = self.width - 2 * crop_pixels_w
            adjusted_height = self.height - 2 * crop_pixels_h
            image_size = (adjusted_width, adjusted_height)
            print(
                f"Using adjusted dimensions for rectification: {image_size} (with {self.crop_percentage}% crop)"
            )
        else:
            image_size = (self.width, self.height)
            print(f"Using original dimensions for rectification: {image_size}")

        # Choose rectification method based on camera type
        is_fisheye = self.calibration_settings.get("is_fisheye", False)
        if is_fisheye:
            print("Both cameras are fisheye - using fisheye rectification")
            return self.rectify_stereo_fisheye(
                cmtx0, dist0, cmtx1, dist1, R, T, image_size
            )
        else:
            print("Both cameras are standard - using standard rectification")
            return self.rectify_stereo_standard(
                cmtx0, dist0, cmtx1, dist1, R, T, image_size
            )

    def verify_rectification(
        self,
        camera0_rect: Dict[str, np.ndarray],
        camera1_rect: Dict[str, np.ndarray],
        camera0_name: str,
        camera1_name: str,
    ) -> None:
        """
        Show live rectified video to verify rectification quality.

        Parameters:
            camera0_rect: Rectification parameters for camera0
            camera1_rect: Rectification parameters for camera1
            camera0_name: Name of first camera
            camera1_name: Name of second camera
        """
        print("Opening cameras to verify rectification...")

        try:
            # Open camera(s)
            if self.use_single_camera:
                cap = self.open_camera(self.camera0, is_stereo=True)
            else:
                cap0 = self.open_camera(self.camera0)
                cap1 = self.open_camera(self.camera1)
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            return

        # Extract rectification maps
        map0_x = camera0_rect["map_x"]
        map0_y = camera0_rect["map_y"]
        map1_x = camera1_rect["map_x"]
        map1_y = camera1_rect["map_y"]

        # Get ROIs if available
        roi0 = camera0_rect.get("roi", None)
        roi1 = camera1_rect.get("roi", None)

        print("Displaying rectified frames. Press ESC to exit.")

        try:
            while True:
                # Get frames
                if self.use_single_camera:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error reading from stereo camera")
                        break
                    frame0, frame1 = split_stereo_frame(frame)
                else:
                    ret0, frame0 = cap0.read()
                    ret1, frame1 = cap1.read()
                    if not ret0 or not ret1:
                        print("Error reading from cameras")
                        break

                # Apply cropping if needed
                if self.crop_percentage > 0:
                    frame0 = crop_frame(frame0, self.crop_percentage)
                    frame1 = crop_frame(frame1, self.crop_percentage)

                # Apply rectification
                frame0_rectified = cv2.remap(frame0, map0_x, map0_y, cv2.INTER_LINEAR)
                frame1_rectified = cv2.remap(frame1, map1_x, map1_y, cv2.INTER_LINEAR)

                # Apply ROI cropping if available
                if roi0 is not None:
                    x, y, w, h = roi0
                    if w > 0 and h > 0:
                        frame0_rectified = frame0_rectified[y : y + h, x : x + w]

                if roi1 is not None:
                    x, y, w, h = roi1
                    if w > 0 and h > 0:
                        frame1_rectified = frame1_rectified[y : y + h, x : x + w]

                # Draw horizontal lines to check rectification
                for i in range(0, self.height, 50):
                    cv2.line(frame0_rectified, (0, i), (self.width, i), (0, 255, 0), 1)
                    cv2.line(frame1_rectified, (0, i), (self.width, i), (0, 255, 0), 1)

                # Create side-by-side display
                combined = np.hstack((frame0_rectified, frame1_rectified))

                # Resize if too large
                max_width = 1600
                if combined.shape[1] > max_width:
                    scale = max_width / combined.shape[1]
                    combined = cv2.resize(combined, None, fx=scale, fy=scale)

                cv2.imshow("Rectified Stereo Pair", combined)

                if cv2.waitKey(1) == 27:  # ESC to exit
                    break
        finally:
            # Release resources
            if self.use_single_camera:
                cap.release()
            else:
                cap0.release()
                cap1.release()
            cv2.destroyAllWindows()

    def save_rectification_maps(
        self,
        camera0_rect: Dict[str, np.ndarray],
        camera1_rect: Dict[str, np.ndarray],
        directory: str = "camera_parameters",
    ) -> None:
        """
        Save rectification maps to files for later use.

        Parameters:
            camera0_rect: Rectification data for camera0
            camera1_rect: Rectification data for camera1
            directory: Output directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save camera0 data
        for name, data in [
            ("camera0_mapx.npy", camera0_rect["map_x"]),
            ("camera0_mapy.npy", camera0_rect["map_y"]),
            ("camera0_rect_R.npy", camera0_rect["R"]),
            ("camera0_rect_P.npy", camera0_rect["P"]),
        ]:
            np.save(os.path.join(directory, name), data)

        # Save camera1 data
        for name, data in [
            ("camera1_mapx.npy", camera1_rect["map_x"]),
            ("camera1_mapy.npy", camera1_rect["map_y"]),
            ("camera1_rect_R.npy", camera1_rect["R"]),
            ("camera1_rect_P.npy", camera1_rect["P"]),
        ]:
            np.save(os.path.join(directory, name), data)

        # Save ROIs if available
        if "roi" in camera0_rect:
            np.save(os.path.join(directory, "camera0_roi.npy"), camera0_rect["roi"])
        if "roi" in camera1_rect:
            np.save(os.path.join(directory, "camera1_roi.npy"), camera1_rect["roi"])

        print(f"Rectification maps and parameters saved to {directory}")

    def get_world_space_origin(
        self, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray, img_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate transformation from world space to camera.

        Parameters:
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Distortion coefficients
            img_path: Path to reference image with checkerboard

        Returns:
            Tuple of rotation matrix and translation vector
        """
        # Read reference image
        frame = cv2.imread(img_path, 1)
        if frame is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Get checkerboard 3D points
        object_points = self.get_checkerboard_object_points()

        # Detect checkerboard
        found, corners = self.detect_checkerboard(frame)
        if not found:
            raise RuntimeError(f"Could not find checkerboard in image: {img_path}")

        # Visualize detected corners
        display_frame = self.display_checkerboard(frame, corners)
        self.put_text(
            display_frame,
            "If you don't see detected points, try with a different image",
            (50, 50),
        )
        cv2.imshow("Reference Image", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Solve PnP to get transformation
        _, rvec, tvec = cv2.solvePnP(
            object_points, corners, camera_matrix, distortion_coeffs
        )
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        return rotation_matrix, tvec

    def get_cam1_to_world_transforms(
        self,
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
        Calculate and visualize transformation from world space to camera1.

        Parameters:
            cmtx0: Camera0 intrinsic matrix
            dist0: Camera0 distortion coefficients
            R_W0: Rotation from world to camera0
            T_W0: Translation from world to camera0
            cmtx1: Camera1 intrinsic matrix
            dist1: Camera1 distortion coefficients
            R_01: Rotation from camera0 to camera1
            T_01: Translation from camera0 to camera1
            image_path0: Path to camera0 reference image
            image_path1: Path to camera1 reference image

        Returns:
            Tuple of rotation and translation from world to camera1
        """
        # Load reference images
        frame0 = cv2.imread(image_path0, 1)
        frame1 = cv2.imread(image_path1, 1)

        if frame0 is None or frame1 is None:
            raise RuntimeError("Could not read reference images")

        # Define unit coordinate axes for visualization
        unit_points = 5 * np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
        ).reshape((4, 1, 3))

        # Colours for XYZ axes
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue

        # Project axes onto camera0 view
        points_projected, _ = cv2.projectPoints(unit_points, R_W0, T_W0, cmtx0, dist0)
        points_2d = points_projected.reshape((4, 2)).astype(np.int32)

        # Draw axes on camera0 image
        origin = tuple(points_2d[0])
        for colour, point in zip(colours, points_2d[1:]):
            point_pixel = tuple(point.astype(np.int32))
            cv2.line(frame0, origin, point_pixel, colour, 2)

        # Calculate camera1-to-world transformation
        R_W1 = R_01 @ R_W0  # Rotation from world to camera1
        T_W1 = R_01 @ T_W0 + T_01  # Translation from world to camera1

        # Project axes onto camera1 view
        points_projected, _ = cv2.projectPoints(unit_points, R_W1, T_W1, cmtx1, dist1)
        points_2d = points_projected.reshape((4, 2)).astype(np.int32)

        # Draw axes on camera1 image
        origin = tuple(points_2d[0])
        for colour, point in zip(colours, points_2d[1:]):
            point_pixel = tuple(point.astype(np.int32))
            cv2.line(frame1, origin, point_pixel, colour, 2)

        # Display results
        cv2.imshow("Camera 0 with world axes", frame0)
        cv2.imshow("Camera 1 with world axes", frame1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return R_W1, T_W1


def modify_calibration_state_for_rectification():
    """Add rectification fields to CalibrationState class."""
    # Extend CalibrationState with rectification parameters
    CalibrationState.rect_R0 = None
    CalibrationState.rect_P0 = None
    CalibrationState.rect_R1 = None
    CalibrationState.rect_P1 = None
    CalibrationState.disparity_to_depth_map = None
    CalibrationState.roi0 = None
    CalibrationState.roi1 = None


def ask_yes_no_question(question: str) -> bool:
    """
    Ask a yes/no question and get response.

    Parameters:
        question: Question text

    Returns:
        True for yes, False for no
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

    stereo_calibrator = StereoCalibrator(sys.argv[1])

    # Check if we're using single camera mode and update settings accordingly
    if stereo_calibrator.use_single_camera:
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
        stereo_calibrator.save_frames_single_camera("camera0")
        # TODO only set to true if frames were actually collected
        calibration_state.frames_collected_cam0 = True
        calibration_state.save()
    else:
        print("Camera 0 frames already collected. Skipping.")

        # Option to recollect if needed
        if ask_yes_no_question("Do you want to recollect frames for camera0?"):
            stereo_calibrator.save_frames_single_camera("camera0")
            calibration_state.frames_collected_cam0 = True
            calibration_state.save()

    # Camera 1 - potentially skip in single camera mode since we already collected both
    if stereo_calibrator.use_single_camera:
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
            stereo_calibrator.save_frames_single_camera("camera1")
            calibration_state.frames_collected_cam1 = True
            calibration_state.save()
        else:
            print("Camera 1 frames already collected. Skipping.")

            # Option to recollect if needed
            if ask_yes_no_question("Do you want to recollect frames for camera1?"):
                stereo_calibrator.save_frames_single_camera("camera1")
                calibration_state.frames_collected_cam1 = True
                calibration_state.save()

    """Step2. Obtain camera intrinsic matrices and save them"""
    print_step_banner(2, "CALIBRATE CAMERA INTRINSICS")

    # Camera 0 intrinsics
    if not calibration_state.intrinsics_calibrated_cam0:
        images_prefix = os.path.join("frames", "camera0*")
        cmtx0, dist0, rmse0 = (
            stereo_calibrator.calibrate_camera_for_intrinsic_parameters(images_prefix)
        )

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
            cmtx0, dist0, rmse0 = (
                stereo_calibrator.calibrate_camera_for_intrinsic_parameters(
                    images_prefix
                )
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
        cmtx1, dist1, rmse1 = (
            stereo_calibrator.calibrate_camera_for_intrinsic_parameters(images_prefix)
        )

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
            cmtx1, dist1, rmse1 = (
                stereo_calibrator.calibrate_camera_for_intrinsic_parameters(
                    images_prefix
                )
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
        stereo_calibrator.save_frames_two_cams(
            "camera0", "camera1"
        ).stereo_frames_collected = True
        calibration_state.save()
    else:
        print("Stereo frames already collected. Skipping.")

        # Option to recollect if needed
        if ask_yes_no_question("Do you want to recollect stereo frames?"):
            stereo_calibrator.save_frames_two_cams("camera0", "camera1")
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

        R1, T1, rmse_stereo = stereo_calibrator.stereo_calibrate(
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
            R1, T1, rmse_stereo = stereo_calibrator.stereo_calibrate(
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
    P0, P1 = stereo_calibrator.check_calibration(
        "camera0", camera0_data, "camera1", camera1_data, _zshift=60.0
    )

    # Update calibration state with projection matrices
    calibration_state.projection_matrix0 = P0
    calibration_state.projection_matrix1 = P1
    calibration_state.save()

    """Step6. Generate rectification maps for stereo vision."""
    print_step_banner(6, "GENERATE RECTIFICATION MAPS")
    # TODO fix rectification

    # Check if rectification is needed
    should_rectify = stereo_calibrator.calibration_settings.get(
        "perform_rectification", False
    )

    if should_rectify and calibration_state.stereo_calibrated:
        print("Generating rectification maps...")

        # Create rectification maps
        camera0_rect, camera1_rect = stereo_calibrator.create_rectification_maps(
            cmtx0, dist0, cmtx1, dist1, R1, T1
        )

        # Save rectification maps for later use
        stereo_calibrator.save_rectification_maps(camera0_rect, camera1_rect)

        stereo_calibrator.verify_rectification(
            camera0_rect, camera1_rect, "camera0", "camera1"
        )
    elif should_rectify:
        print("Cannot generate rectification maps: stereo calibration not completed")
    else:
        print("Rectification step skipped (not enabled in settings)")

    """Optional. Define a different origin point and save the calibration data"""
    print_step_banner(7, "OPTIONAL: DEFINE ALTERNATIVE WORLD ORIGIN (OPTIONAL)")

    if ask_yes_no_question("Do you want to define an alternative world origin?"):
        print("Define a different world space origin using a checkerboard...")
        # Get the world to camera0 rotation and translation
        R_W0, T_W0 = stereo_calibrator.get_world_space_origin(
            cmtx0, dist0, os.path.join("frames_pair", "camera0_4.png")
        )
        # Get rotation and translation from world directly to camera1
        R_W1, T_W1 = stereo_calibrator.get_cam1_to_world_transforms(
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
        P_W0 = stereo_calibrator.get_projection_matrix(cmtx0, R_W0, T_W0)
        P_W1 = stereo_calibrator.get_projection_matrix(cmtx1, R_W1, T_W1)

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
