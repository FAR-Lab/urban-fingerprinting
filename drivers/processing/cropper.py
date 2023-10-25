import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.processing.Cropper import Cropper

if __name__ == '__main__': 
    DoC = sys.argv[1]
    image_dir = sys.argv[2]

    cropper = Cropper(DoC, image_dir)
    cropper()