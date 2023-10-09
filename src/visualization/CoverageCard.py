# FARLAB - UrbanECG 
# Developer: Matt Franchi, with help from GitHub CoPilot 
# Last Modified: 10/08/2023

# This script contains the CoverageCard class, which is used to create a coverage card plot for a given map. 

# Imports 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib as mpl

import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.analysis.DayOfCoverage import DayOfCoverage

import datetime


# CoverageCard class
class CoverageCard(): 
    def __init__(self, days_of_coverage): 
        # CASE: list of Y-m-d strings
        if all(map(lambda x: isinstance(x, str), days_of_coverage)):
            # convert strings to datetimes
            self.days_of_coverage = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), days_of_coverage))
        # CASE: list of datetime objects
        elif all(map(lambda x: isinstance(x, datetime.datetime), days_of_coverage)):
            self.days_of_coverage = days_of_coverage
        # CASE: list of DayOfCoverage objects
        elif all(map(lambda x: isinstance(x, DayOfCoverage), days_of_coverage)):
            self.days_of_coverage = list(map(lambda x: x.date, days_of_coverage))
            # convert date strings to datetime objects 
            self.days_of_coverage = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), self.days_of_coverage))
        else:
            raise ValueError('days_of_coverage must be a list of strings, datetimes, or DayOfCoverage objects')

        # sort days of coverage
        self.days_of_coverage.sort()

        # calculate time delta between earliest and latest days of coverage
        self.delta = self.days_of_coverage[-1] - self.days_of_coverage[0]
    
    def latex(self, latex=True):
        """
        Toggle latex rendering. 
        """
        if latex: 
            mpl.rcParams['text.usetex'] = True
        else: 
            mpl.rcParams['text.usetex'] = False

    def plot(self): 
        """
        Generate calendar plot of coverage for the given days. 
        """
        # if delta within a week, plot within week
        if self.delta.days <= 7: 
            self.plot_within_week()
        # if delta within a month, plot within month
        elif self.delta.days <= 30: 
            self.plot_within_month()
        # if delta within a year, plot within year
        elif self.delta.days <= 365: 
            self.plot_within_year()
        



    
    def plot_within_week(self): 

        padding_h = 0.05 
        padding_v = 0.05

        box_height = 0.5 

        fig, ax = plt.subplots(figsize=(9, 3))
        # set axis limits
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 1)

        ax.set_frame_on(False)
        
        # set axis ticks
        ax.set_xticks(np.arange(7))
        ax.set_yticks([])
        # set axis tick labels based on delta
        # get earliest day of coverage
        earliest_day = self.days_of_coverage[0]
        ax.set_xticklabels([(earliest_day + datetime.timedelta(days=x)).strftime("%a %m-%d-%y") for x in range(7)])
        # format xtick labels to Y-M-D 
        xticklabels = ax.get_xticklabels()
       

        # have xtick labels in between each day 
        ax.xaxis.set_ticks_position('bottom')

        # set axis labels
        ax.set_xlabel('Day of Week')
        # set title
        ax.set_title('Coverage Card')
        # plot coverage
        for day in [earliest_day + datetime.timedelta(days=x) for x in range(7)]:
            if day in self.days_of_coverage:
                ax.axvspan(day.weekday() + padding_h, day.weekday() + (1-padding_h), ymin=padding_v, ymax=box_height-padding_v, facecolor='green', alpha=0.5, capstyle='round')
            else:
                ax.axvspan(day.weekday() + padding_h , day.weekday() + (1-padding_h), ymin=padding_v, ymax=box_height-padding_v, facecolor='red', alpha=0.5)
        
        # add whitespace padding at bottom of plot 
        plt.subplots_adjust(bottom=0.2)
        


        # plot legend
        green_patch = mpatches.Patch(color='green', label='Coverage')
        red_patch = mpatches.Patch(color='red', label='Missing Coverage')
        plt.legend(handles=[green_patch, red_patch])

        plt.savefig('coverage_card.png')



    def plot_within_month(self):
        pass

    def plot_within_year(self):
        pass




