import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import yaml
from scipy import linalg


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

    # Calibration quality (RMSE)
    rmse_camera0: Optional[float] = None
    rmse_camera1: Optional[float] = None
    rmse_stereo: Optional[float] = None

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


def parse_calibration_settings_file(filename: str) -> None:
    """
    Parse and load calibration settings from a YAML configuration file.

    Parameters:
        filename: Path to the calibration settings YAML file

    Returns:
        dict: Calibration settings loaded from the file

    Raises:
        SystemExit: If file doesn't exist or doesn't contain expected format
    """
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
    return calibration_settings


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
