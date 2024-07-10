# Small-scale, Custom Instrumentation (SSCI) Paradigm

## Data 
This approach utilizes visual data taken at a smaller-scale, using consumer devices like GoPro action cameras. With a lack of standardization that the LSCA paradigm provides, there is much more leeway to modify the data processing and analysis pipeline to fit the needs of your project. Nonetheless, we provide details on how to replicate the approach used to gather data for our lab's cross-cultural driving & pedestrian behavior studies. 


### Sample Data: NYC GoPro Footage 
From FARLAB's cross-cultural pedestrian behavior work, we provide a short sample of footage taken by a participant during their daily commute. This footage is recorded with a GoPro action camera, producing high-resolution 4K video. As mentioned in our paper, gathering accurate telemetry data using only a GoPro is infeasible; we provide instructions for replicating our OpenHaystack-based Apple Airtag solution in the [XXX](LINK) section. Sample footage is available at [this](https://cornell.box.com/v/urban-fingerprinting-sample) link, through Box. 


## Processing 
We provide code to do the following (although, note as we only explored this paradigm, the code is not particularly robust). 
1. Detect pedestrians, vehicles, and other objects of interest in SSCI-compliant footage. 
2. Merge and aggregate counts of detected objects at the *clip* level, or over time. 
    - NOTE: We only develop a proof-of-concept of our AirTag localization, and so we omit the granular geographic localization developed for the LSCA paradigm. 

We will describe each step in detail.

### 1. Object Detection
Need to find this script from the repo somewhere, its' not currently in UrbanECG, but somewhere else. 

### 2. Aggregation 
Also need to find this script from whatever repo it is in.




## A. Geographic Localization with AirTag
