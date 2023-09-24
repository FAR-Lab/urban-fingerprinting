class DayOfCoverage:
    """
    A class to represent a day of coverage from the Nexar dataset. 

    ...

    Attributes 
    ----------
    date : str
        The date of the day of coverage, in the format YYYY-MM-DD
    frames_data : pd.DataFrame
        A pandas dataframe containing the metadata for all frames in the day of coverage.
    nearest_edges : pd.DataFrame
        A pandas dataframe containing the nearest edges to each frame in frames_data.
    nearest_edges_dist : pd.DataFrame
        A pandas dataframe containing the distance to the nearest edge for each frame in frames_data.
    
    Methods
    -------
    None
    """
    def __init__(self, day_of_coverage):
        self.date = day_of_coverage 
        self.frames_data = [] 
        self.nearest_edges = [] 
        self.nearest_edges_dist = []
        self.detections = []