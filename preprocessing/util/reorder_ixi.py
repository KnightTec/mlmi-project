import os
import shutil
import sys

# The list of modalities
modalities = ['IXI-PD', 'IXI-T1', 'IXI-T2', 'IXI-MRA']

def get_session_name(filename):
    """Extract the session name (e.g., IXI002-Guys-0828) from the filename"""
    parts = filename.split('-')
    if len(parts) > 3:
        return '-'.join(parts[:3])
    return None

def main():
    # Directory containing the modality folders (e.g., IXI-PD, IXI-T1, etc.)
    root_dir = sys.argv[1]
    sessions_dir = sys.argv[2]

    # Ensure the sessions directory exists
    os.makedirs(sessions_dir, exist_ok=True)

    for modality in modalities:
        modality_path = os.path.join(root_dir, modality)
        
        if os.path.exists(modality_path):
            for filename in os.listdir(modality_path):
                # Ensure the file has the correct extension
                if filename.endswith(".nii.gz"):
                    session_id = get_session_name(filename)
                    
                    # If session ID is found, create a directory and move files
                    if session_id:
                        session_path = os.path.join(sessions_dir, session_id)
                        os.makedirs(session_path, exist_ok=True)
                        
                        src_path = os.path.join(modality_path, filename)
                        dst_path = os.path.join(session_path, filename)
                        
                        shutil.move(src_path, dst_path)
                        print(f"Moved {src_path} to {dst_path}")

if __name__ == "__main__":
    main()
