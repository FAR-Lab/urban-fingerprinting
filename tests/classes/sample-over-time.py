import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.visualization.SampleOverTime import SampleOverTime

if __name__ == "__main__": 
    sample_csv_path = sys.argv[1]
    md_tld = sys.argv[2]

    sample_over_time = SampleOverTime(sample_csv_path, md_tld)
    sample_over_time()