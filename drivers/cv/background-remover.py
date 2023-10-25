import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cv.BackgroundRemover import BackgroundRemover

if __name__ == '__main__':
    dir_to_scan = sys.argv[1]
    output_dir = sys.argv[2]

    # make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    background_remover = BackgroundRemover()
    background_remover.batch(dir_to_scan, output_dir)