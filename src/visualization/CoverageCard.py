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

from pathlib import Path

import calendar 

import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.analysis.DayOfCoverage import DayOfCoverage

import datetime


# CoverageCard class
class CoverageCard(): 
    def __init__(self, days_of_coverage, attached_plot=""): 
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

        self.attached_plot = attached_plot

        self.cosmetics = {
            "covered_color": "seagreen",
            "missing_color": "linen",
            "covered_alpha": 0.5,
            "missing_alpha": 0.5,

        }

    
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



    def plot_within_month(self, data = None, ax = None, save=True):

        if data is None: 
            data = self.days_of_coverage
        else: 
            data = data

        earliest_day = data[0]

        # set up figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = plt.gcf()

        box_height = 0.5 
        box_width = 0.5 

        outer_padding_h = 0.05
        outer_padding_v = 0.05

        # set axis limits
        ax.set_xlim(0, 7*box_width + 2*outer_padding_h)
        ax.set_ylim(0, 6*box_height + 2*outer_padding_v)

        ax.set_frame_on(False)

        # no axis ticks 
        ax.set_xticks([])
        ax.set_yticks([])

        # get day of week for first day of month
        first_of_month = earliest_day.replace(day=1)
        first_of_month_day = first_of_month.weekday()

        # get number of days in month 
        num_days_in_month = calendar.monthrange(earliest_day.year, earliest_day.month)[1]
        for day in [first_of_month + datetime.timedelta(days=x) for x in range(num_days_in_month)]:
            row = 5 - (day.day + first_of_month_day) // 7
            # handle case where first day of month is a Sunday
            if first_of_month_day == 6:
                row += 1
            


            col = (day.weekday() + 1) % 7

            if day in data:
                ax.add_patch(mpatches.Rectangle((col*box_width + outer_padding_h, row*box_height + outer_padding_v), box_width, box_height, facecolor=self.cosmetics['covered_color'], alpha=self.cosmetics['covered_alpha'], capstyle='round'))
                # add shadow around patch 
                ax.add_patch(mpatches.Rectangle((col*box_width + outer_padding_h, row*box_height + outer_padding_v), box_width, box_height, facecolor='none', edgecolor='grey', linewidth=2, alpha=0.5, capstyle='round')) 
                
                # add label inplace 
                ax.text(col*box_width + outer_padding_h + box_width/2, 
                        row*box_height + outer_padding_v + box_height/2, 
                        day.day, horizontalalignment='center', verticalalignment='center')
            else:
                ax.add_patch(mpatches.Rectangle((col*box_width + outer_padding_h, row*box_height + outer_padding_v), box_width, box_height, facecolor=self.cosmetics['missing_color'], alpha=self.cosmetics['missing_alpha'], capstyle='round'))
                ax.add_patch(mpatches.Rectangle((col*box_width + outer_padding_h, row*box_height + outer_padding_v), box_width, box_height, facecolor='none', edgecolor='grey', linewidth=1.5, alpha=0.5, capstyle='round')) 
                # add label inplace
                ax.text(col*box_width + outer_padding_h + box_width/2, 
                        row*box_height + outer_padding_v + box_height/2, 
                        day.day, horizontalalignment='center', verticalalignment='center')
                

        # plot legend
        green_patch = mpatches.Patch(color=self.cosmetics['covered_color'], label='Day is in Plot')
        red_patch = mpatches.Patch(color=self.cosmetics['missing_color'], label='Day Excluded')
        plt.legend(handles=[green_patch, red_patch], loc='lower right', 
                   fancybox=True, framealpha=1)
        # move legend down
      

        # add title with month and year 
        if self.attached_plot != "": 
            ax.set_title(self.attached_plot + "\n" + earliest_day.strftime("%B %Y")+'\n\n')
        else:
            ax.set_title(earliest_day.strftime("%B %Y")+"\n")

        
        
        
        # add days of week labels below title, and above the first row of data 
        for i in range(7):
            i_day = (i-1) % 7
            ax.text(i*box_width + outer_padding_h + box_width/2, 5.65*box_height + outer_padding_v + box_height/2, calendar.day_abbr[i_day], horizontalalignment='center', verticalalignment='center')

    
        if save:
            plt.savefig(f'{Path(self.attached_plot).stem}_coverage_card.png', bbox_inches='tight')

        return ax





    def plot_within_year(self):
    

        # split days of coverage into months
        months = {}
        for day in self.days_of_coverage:
            month = day.strftime('%B %Y')
            if month not in months:
                months[month] = []
            months[month].append(day)
        
        # plot each month
        # number of subplots = number of months 
        num_subplots = len(months.keys())

        # figure out optimal number of rows and columns
        num_rows = 1
        num_cols = 1
        while num_rows * num_cols < num_subplots:
            if num_rows == num_cols:
                num_cols += 1
            else:
                num_rows += 1

        # set number of rows and columns 
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 18))
        # turn off frames 
        for ax in axs.flat:
            ax.set_frame_on(False)
        
        # turn off axis 
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        

        # plot each month in corresponding subplot 
        for idx, month in enumerate(months.keys()):
            days = months[month]
            # plot month in corresponding subplot 
            plot_idx = np.unravel_index(idx, axs.shape)
            
            # add month plot to corresponding subplot
            self.plot_within_month(days, axs[plot_idx], save=False)
        

        plt.savefig(f'{Path(self.attached_plot).stem}_coverage_card.png')





