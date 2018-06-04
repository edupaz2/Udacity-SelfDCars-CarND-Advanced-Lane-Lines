# Define a class to receive the characteristics of each line detection
import numpy as np
from utils import ym_per_pix, xm_per_pix

class Line():
    def __init__(self, width=1280, height=720, n=10):
        # last n iterations
        self.n = n
        self.w = width
        self.h = height
        self.ploty = np.linspace(0, self.h-1, self.h)
        self.ploty2 = self.ploty**2

        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line.
        self.recent_xfitted = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
    # Adds a new fit
    def add_fit(self, fit):
        #fit = np.polyfit(y, x, 2)
        fitx = fit[0]*self.ploty2 + fit[1]*self.ploty + fit[2]
        if fit is not None:
            if not self.detected or self.best_fit is None:
                self.detected = True
            else:
                # Check diff between fit and best fit
                self.diffs = abs(self.best_fit - fit)
                if self.diffs[0] < 0.01 and self.diffs[1] < 1.0 and self.diffs[2] < 100.0:
                    # We have a good fit.
                    self.detected = True
                else:
                    # We have a bad fit
                    self.detected = False
        else:
            self.detected = False

        if self.detected == True:
            self.recent_xfitted.append(fitx[-1])
            self.recent_xfitted = self.recent_xfitted[-self.n:]# Only keep n
            
            self.current_fit.append(fit)
            self.current_fit = self.current_fit[-self.n:]# Only keep n
            self.best_fit = np.average(self.current_fit, axis=0)

            # Fit new polynomials to x,y in world space
            fit_cr = np.polyfit(self.ploty*ym_per_pix, fitx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            y_eval = self.ploty[-1]
            self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
            self.line_base_pos = self.recent_xfitted[-1]

