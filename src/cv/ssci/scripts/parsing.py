import glob
import fire
import numpy as np
import cv2
import os 


NUM_FRAMES_TO_AVG = 3

# Number of frames to pad the intersection region with (before and after)
TIME_OFFSET = 60

def merge_periods(tuples):
    print(tuples)
    sorted_tuples = sorted(tuples)
    merged = [sorted_tuples[0]]
    
    for current in sorted_tuples[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    
    return merged

def parse(source, base, TOP_LEVEL_DIR, PROCESSED_SUFFIX):

    footage = cv2.VideoCapture(f"{source}")
    num_frames = int(footage.get(cv2.CAP_PROP_FRAME_COUNT))

    del footage

    crosswalk_detections = [] 
    coco_detections = [] 

    for i in range(num_frames): 
        crosswalk_detections.append(f"{base}/{TOP_LEVEL_DIR}/{os.path.splitext(os.path.basename(source))[0]}_cdnet/labels/{os.path.splitext(os.path.basename(source))[0]}_{PROCESSED_SUFFIX}_{i}.txt")
        coco_detections.append(f"{base}/{TOP_LEVEL_DIR}/{os.path.splitext(os.path.basename(source))[0]}_coco/labels/{os.path.splitext(os.path.basename(source))[0]}_{PROCESSED_SUFFIX}_{i}.txt")





    intersection_frames = np.zeros(num_frames, dtype=bool)
    assert num_frames == len(coco_detections)
    i = 0
    for i in range(0, num_frames, NUM_FRAMES_TO_AVG):
        avg_crosswalk_detections = []
        avg_coco_detections = []

        for j in range(NUM_FRAMES_TO_AVG):
            try: 
                #print(crosswalk_detections[i+j])
                with open(crosswalk_detections[i+j], 'r', errors='ignore') as crosswalk_file:
                    for line in crosswalk_file:
                        line = line.strip().split()
                        if len(line) > 0:
                            avg_crosswalk_detections.append(line)
            except: 
                pass
                #print(f"Error at frame {i+j}; crosswalk")
            
            try: 
                with open(coco_detections[i+j], 'r', errors='ignore') as coco_file:
                    for line in coco_file:
                        line = line.strip().split()
                        if len(line) > 0:
                            avg_coco_detections.append(line)
                    
            except: 
                pass
                #print(f"Error at frame {i+j}; coco")
            

        if (len(avg_crosswalk_detections) + len(avg_coco_detections)) > (NUM_FRAMES_TO_AVG):
            #print(f"Frame {i} has {len(avg_crosswalk_detections)} crosswalk detections and {len(avg_coco_detections)} coco detections")
            intersection_frames[i:i+NUM_FRAMES_TO_AVG] = True

    # At this point, intersection frames should be identified
    # We need to return periods of the overall video that contain intersections
    # We can do this by finding the first and last intersection frames
    # and then padding them by TIME_OFFSET seconds
    intersection_frames = np.where(intersection_frames)[0]
    periods = []

    if len(intersection_frames) > 0:
        start_frame = intersection_frames[0]
        end_frame = intersection_frames[0]

        for frame in intersection_frames[1:]:
            if frame == end_frame + 1:
                end_frame = frame
            else:
                periods.append((start_frame, (end_frame + 1) ))
                start_frame = frame
                end_frame = frame

        periods.append((start_frame, (end_frame + 1)))


        padded_periods = []
        for start_time, end_time in periods:
            padded_start_time = max(0, start_time - TIME_OFFSET)
            padded_end_time = min(num_frames,end_time + TIME_OFFSET)
            padded_periods.append((padded_start_time, padded_end_time))


        print(padded_periods)

        return merge_periods(padded_periods)
    

    return None

if __name__ == '__main__':
    fire.Fire(parse)
    
