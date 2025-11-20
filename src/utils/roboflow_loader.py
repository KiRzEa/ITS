"""
Roboflow Dataset Loader
Downloads and manages traffic sign detection dataset from Roboflow
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from roboflow import Roboflow
import yaml
import json


class RoboflowDataLoader:
    """Load and manage dataset from Roboflow API"""

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int = 1,
        data_dir: str = "data/raw"
    ):
        """
        Initialize Roboflow data loader

        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project ID
            version: Dataset version number
            data_dir: Directory to save downloaded data
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Roboflow client
        self.rf = Roboflow(api_key=api_key)
        self.workspace_obj = self.rf.workspace(workspace)
        self.project_obj = self.workspace_obj.project(project)
        self.dataset = None

    def download_dataset(
        self,
        format: str = "yolov8",
        location: Optional[str] = None
    ) -> str:
        """
        Download dataset from Roboflow

        Args:
            format: Dataset format ('yolov8', 'yolov5', 'coco', 'voc', etc.)
            location: Custom download location (optional)

        Returns:
            Path to downloaded dataset
        """
        if location is None:
            location = str(self.data_dir / format)

        print(f"Downloading dataset in {format} format...")
        print(f"Workspace: {self.workspace}")
        print(f"Project: {self.project}")
        print(f"Version: {self.version}")

        try:
            version_obj = self.project_obj.version(self.version)
            self.dataset = version_obj.download(
                model_format=format,
                location=location
            )

            print(f"Dataset downloaded successfully to: {self.dataset.location}")

            # Save dataset info
            self._save_dataset_info(format)

            return self.dataset.location

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not downloaded yet. Call download_dataset() first.")

        info = {
            "workspace": self.workspace,
            "project": self.project,
            "version": self.version,
            "location": self.dataset.location,
        }

        # Try to read data.yaml if it exists (YOLO format)
        yaml_path = Path(self.dataset.location) / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                info.update({
                    "num_classes": yaml_data.get('nc', 0),
                    "class_names": yaml_data.get('names', []),
                    "train_path": yaml_data.get('train', ''),
                    "val_path": yaml_data.get('val', ''),
                    "test_path": yaml_data.get('test', ''),
                })

        return info

    def _save_dataset_info(self, format: str):
        """Save dataset information to JSON file"""
        info = {
            "workspace": self.workspace,
            "project": self.project,
            "version": self.version,
            "format": format,
            "location": str(self.dataset.location),
        }

        # Try to read additional info from data.yaml
        yaml_path = Path(self.dataset.location) / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                info.update({
                    "num_classes": yaml_data.get('nc', 0),
                    "class_names": yaml_data.get('names', []),
                })

        # Save to JSON
        info_path = self.data_dir / f"dataset_info_{format}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Dataset info saved to: {info_path}")

    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        info = self.get_dataset_info()
        return info.get('class_names', [])

    def get_num_classes(self) -> int:
        """Get number of classes"""
        info = self.get_dataset_info()
        return info.get('num_classes', 0)


def load_from_env(env_path: str = ".env") -> RoboflowDataLoader:
    """
    Load Roboflow data loader from environment file

    Args:
        env_path: Path to .env file

    Returns:
        RoboflowDataLoader instance
    """
    from dotenv import load_dotenv

    load_dotenv(env_path)

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project = os.getenv("ROBOFLOW_PROJECT")
    version = int(os.getenv("ROBOFLOW_VERSION", 1))

    if not all([api_key, workspace, project]):
        raise ValueError("Missing required environment variables. Check your .env file.")

    return RoboflowDataLoader(
        api_key=api_key,
        workspace=workspace,
        project=project,
        version=version
    )


if __name__ == "__main__":
    # Example usage
    loader = load_from_env()

    # Download in multiple formats
    for format in ["yolov8", "coco"]:
        print(f"\n{'='*50}")
        print(f"Downloading {format} format")
        print(f"{'='*50}")
        loader.download_dataset(format=format)

    # Print dataset info
    print("\n" + "="*50)
    print("Dataset Information")
    print("="*50)
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
