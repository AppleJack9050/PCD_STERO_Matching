import os
from os import path as osp
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from utils import extract_pose, extract_cam, read_img

class eth3d_dataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None):
        super().__init__()
        # Set the main directory for the dataset and define subdirectories.
        self.root_dir = root_dir
        self.img_dir = osp.join(root_dir, 'images', "dslr_images_undistorted")
        # List all filenames in the image directory.
        self.file_names = os.listdir(self.img_dir)
        # Define the file path for the point cloud data.
        self.pcd_dir = osp.join(root_dir, 'scan_clean', "scan1.ply")

        # Build a list of complete image paths for files ending with '.JPG'.
        self.img_files = [osp.join(root_dir, file) 
                          for file in os.listdir(root_dir) if file.endswith('.JPG')]
        
        # Load calibration data: 6D pose information and camera intrinsic parameters.
        self.pose_df = extract_pose(osp.join(root_dir, 'dslr_calibration_undistorted', 'images.txt'))
        self.cam_df = extract_cam(osp.join(root_dir, 'dslr_calibration_undistorted', 'cameras.txt'))
               
        # Read the point cloud file and convert its points and color attributes to tensors.
        self.pcd = o3d.io.read_point_cloud(self.pcd_dir)
        self.pcd_points = torch.transpose(torch.from_numpy(np.asarray(self.pcd.points), dtype=torch.float32), 1, 0)
        self.pcd_feats = torch.transpose(torch.from_numpy(np.asarray(self.pcd.colors), dtype=torch.float32), 1, 0)
        
    def __len__(self):
        # Return the number of images in the dataset.
        return len(self.file_names)

    def __getitem__(self, idx):
        # Construct the full image path for the given index.
        img_path = osp.join(self.img_dir, self.file_names[idx])
        # Read and resize the image; no augmentation is applied.
        img_data = read_img(img_path, resize=(640, 480), augment_fn=None)
        # Retrieve the 6D pose information for this image as a NumPy array.
        img_pose = self.pose_df.loc[self.file_names[idx], ['QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ']].to_numpy()
        # Extract camera intrinsic parameters (fx, fy, cx, cy) using the CameraID from the pose data.
        cam_param = self.cam_df.loc[self.pose_df.loc[self.file_names[idx]]['CAMERA_ID'], ['fx', 'fy', 'cx', 'cy']].to_numpy()

        # Create a dictionary combining image data and point cloud information.
        data = {
            'img': {
                'feats': img_data,           # Image tensor with shape (3, 640, 480)
                '6d_pose': img_pose,         # 6D pose data with shape (7, )
                'cam_param': cam_param,      # Camera parameters with shape (4, )
            },
            'pcd': {
                'points': self.pcd_points, # Tensor of point cloud coordinates with shape (3, n)
                'feats': self.pcd_feats,   # Tensor of point cloud color features with shape (3, n)
            } 
        }

        return data