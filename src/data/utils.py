import pandas as pd
import torch
import cv2

import pandas as pd

def extract_pose(file_path):
    """
    Extracts pose data from a file and returns it as a pandas DataFrame with IMAGE_ID as the index.
    
    The file is expected to have lines with comments (starting with '#') 
    or blank lines, and pose data lines with 10 whitespace-separated tokens:
      IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      
    This function also removes the substring "dslr_images_undistorted/" from the NAME field.
    
    Parameters:
      file_path: str, path to the file containing the image data.
    
    Returns:
      A pandas DataFrame with the pose data, indexed by IMAGE_ID.
    """
    pose_data = []
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            # Skip header, comments, and empty lines.
            if line.startswith("#") or not line:
                continue
            
            tokens = line.split()
            # Check if the line corresponds to a pose data line (10 tokens).
            if len(tokens) == 10:
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = tokens
                # Remove the unwanted substring from the filename.
                name = name.replace("dslr_images_undistorted/", "")
                pose_data.append({
                    "IMAGE_ID": int(image_id),
                    "QW": float(qw),
                    "QX": float(qx),
                    "QY": float(qy),
                    "QZ": float(qz),
                    "TX": float(tx),
                    "TY": float(ty),
                    "TZ": float(tz),
                    "CAMERA_ID": int(camera_id),
                    "NAME": name
                })
    
    # Create a DataFrame from the pose data.
    df_pose = pd.DataFrame(pose_data)
    # Set IMAGE_ID as the DataFrame index.
    df_pose.set_index('NAME', inplace=True)
    
    return df_pose

def extract_cam(filepath):
    """
    Reads a camera configuration file and returns a DataFrame with camera data.
    
    The file should have lines starting with '#' as comments, and the data should be
    formatted as:
    
      CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
    
    Parameters:
    -----------
    filepath : str
        The path to the camera configuration file.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with CAMERA_ID as the index and columns: MODEL, WIDTH, HEIGHT, fx, fy, cx, cy.
    """
    # Read the file ignoring comment lines (lines starting with '#')
    df = pd.read_csv(filepath, comment='#', header=None, sep=r'\s+')
    
    # Rename the columns
    df.columns = ['CAMERA_ID', 'MODEL', 'WIDTH', 'HEIGHT', 'fx', 'fy', 'cx', 'cy']
    
    # Set CAMERA_ID as the index
    df.set_index('CAMERA_ID', inplace=True)
    
    return df

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = cv2.imread(str(path), )
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image