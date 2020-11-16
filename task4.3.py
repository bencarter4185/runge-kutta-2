# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:20:17 2020

@author: benca
"""

# Clear items in workspace
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import simps

'''
Solution:
    
    for 0 < x <= a:
        -hbar^2/2m d^2/dx^2(phi) - (hbar^2 V0)/(2ma^2) phi = E phi
        
        Let x = ay and lambda = E(2ma^2)/hbar
        
        d^2/dy^2(phi) = - (lambda + V0) phi = f(phi, dphi/dy, y)
    
    for x > a:
        as before
        
        d^2/dy^2(phi) = - (lambda) phi = f(phi, dphi/dy, y)
    
    2ODE, let:
        v = d/dy(phi)
        dv/dy = f(phi, v, y)
        
Will need to incorporate potential as a piecewise function
'''

def f(phi, lam, pot): 
    # Return f(phi, v, y) as set out above; potential defined as a piecewise
    #   function in rk2_sch()
    return - phi * (lam + pot)
    
def rk2_sch(y0, phi0, v0, lam, y1, dy, N_rk, pot):
    # Calculate second order Runge-Kutta for the Schroedinger equation
    
    # Create empty arrays for phi and v
    phi = np.zeros(N_rk+1); v = np.zeros(N_rk+1)
    
    # Place boundary conditions into arrays
    phi[0] = phi0; v[0] = v0
    
    # Iterate through from min y to max y
    for i in np.arange(0, N_rk):
        
        # Read off current phi, v, potential from arrays
        phi_i = phi[i]; vi = v[i]; pot_i = pot[i]
        
        # Calculate new half steps, phi_h, vh
        phi_h = phi_i + dy/2 * vi  
        vh = vi + dy/2 * f(phi = phi_i, lam = lam, pot = pot_i)        
        
        # Update new values from the full step
        phi[i+1] = phi_i + dy * vh
        v[i+1] = vi + dy * f(phi = phi_h, lam = lam, pot = pot_i)
    
    return phi

def shooting_method(y0, v0, phi0, y1, N_rk, N_loops, lam, d_lam_0):
    # Applies a shooting method solution for the Schroedinger equation
    
    # Calculate step size in y
    dy = (y1 - y0)/N_rk
    
    # Generate y
    #
    # Using explicit linspace prevents floating point errors e.g. when using
    #   np.arange()
    start = y0
    stop = y1
    step = dy
    num = 1 + int((stop-start)/step)
    y = np.linspace(start, stop, num, endpoint=True)
    
    # Define potential as a piecewise step function
    #
    # V0 is defined to be 10 in the question, but 0 potential when x > a,
    #   in this case y > 1 as x = ay
    pot = np.piecewise(y, [(y > 0) & (y <= 1), y > 1],
                       [lambda y: 10, lambda y: 0])
    
    
    # Counting variables of the number of iterations
    i = 0
    j = 0
    
    # Variable to denote what value the wavefunction is tending to
    val_tending_to = np.Inf
    
    # Arrays to store the trialled values of lambda; and the calculated values
    #   phi has tended to, as well as the gradient at that point
    lam_vals = np.zeros(N_loops)
    tending_vals = np.zeros(N_loops)
    grad_vals = np.zeros(N_loops)
    
    # Iterate lambda until a solution to the ground state is found or it
    #   times out
    while i < N_loops:
            
        # Calculate phi using the runge-kutta function
        phi = rk2_sch(y0, phi0, v0, lam, y1, dy, N_rk, pot)
        
        # Calculate current gradient (at the end of phi) and average of the
        #   last 10 points    
        grad = (np.gradient(phi))[-1]
        val_tending_to = np.mean(phi[-10:])
        
        # Store these values
        lam_vals[i] = lam
        tending_vals[i] = val_tending_to
        grad_vals[i] = grad
            
        
        # If the current value phi is tending to is of a different sign to
        #   the previous iteration, then apply a bisection method within
        #   a separate while loop
        if np.sign(tending_vals[i]) != np.sign(tending_vals[i-1]) and i > 1:
            
            # Assign a to be the previous value calculated, and b to the
            #   current value
            # Assign the values to be long doubles, in case standard float
            #   doesn't have enough digits
            a = i-1
            b = i
            
            lam_a = np.longdouble(lam_vals[a])
            lam_b = np.longdouble(lam_vals[b])
            val_tending_to_a = np.longdouble(tending_vals[a])
            
            # Use to check for precision errors
            precision_error = False
            
            while j < N_loops:
                # Iterate through and define new midpoints depending on the
                #   sign of the new point calculated
                
                # Define new midpoint, c
                lam_c = np.longdouble((lam_a + lam_b)/2)
                
                # Calculate Runge-Kutta based on this new value
                phi = rk2_sch(y0, phi0, v0, lam_c, y1, dy, N_rk, pot)
                               
                # Calculate new gradients and tending value due to this 
                #   midpoint
                grad_c = np.longdouble((np.gradient(phi))[-1])
                val_tending_to_c = np.longdouble(np.mean(phi[-10:]))
                
                # If phi is tending to zero and the gradient is tending to 
                #   zero, it is a solution
                if np.abs(val_tending_to_c) < tol and np.abs(grad_c) < tol:
                    return lam, y, phi
                
                # If phi (calculated using new midpoint lambda) tends to a
                #   value of the same sign as a, set a to c
                # If different sign, set b to c
                if np.sign(val_tending_to_c) == np.sign(val_tending_to_a):
                    # If setting a to be a value it already is: we have a
                    #   precision error
                    if lam_a == lam_c:
                        precision_error = True                    
                    lam_a = np.longdouble(lam_c)
                    val_tending_to_a = np.longdouble(val_tending_to_c)
                else:
                    # If setting b to be a value it already is: we have a
                    #   precision error
                    if lam_b == lam_c:
                        precision_error = True  
                    lam_b = np.longdouble(lam_c)
                    
                    
                # If precision_error has been set to True: while loop will run
                #   until max number of iterations whilst not getting any
                #   closer to the solution
                # 
                # This is because np.longdouble cannot get any more precise
                # 
                # This error will occur at y1 > 14 for tol = 1e-4
                if precision_error:
                    msg = "".join(["Hit a precision error. Cannot calculate",
                                    " the ground state energy as lambda cannot",
                                    " be set any finer. Please use a smaller",
                                    " value of y1 or a larger tolerance value",
                                    " and try again."])
                    print(msg)
                    sys.exit()
                
                # Iterate the counting variable j
                j += 1
            
            else:
                # If this code is reached: the bisection method while loop has
                #   timed out. Print this to screen and quit the script
                
                msg = "".join(["The code has timed out during the bisection",
                               " method loop. Number of loops is currently set",
                               " to ", str(N_loops), ". Please try again with",
                               " different parameters."])
                print(msg)
                sys.exit()
                        
        # If the wavefunction psi is tending to positive infinity, then
        #   lambda is too negative
        # If tending to negative infinity, then lambda isn't big enough
        if val_tending_to > 0:
            lam += d_lam_0
        else:
            lam -= d_lam_0
            
        # If phi is tending to zero and the gradient is tending to zero, it is
        #   a solution
        # Unlikely to find a solution before dipping into the bisection loop;
        #   but check anyway
        if np.abs(val_tending_to) < tol and np.abs(grad) < tol:
            return lam, y, phi
    
        # Iterate the counting variable i
        i += 1      
    
    else:
        # If this code is reached: the while loop has timed out before hitting
        #   the bisection loop. Print this to screen and quit the script
        
        msg = "".join(["The code has timed out before starting the bisection",
                       " method loop. Number of loops is currently set to ", 
                       str(N_loops),
                       ". Please try again with different parameters."])
        print(msg)
        sys.exit()

# =============================================================================
# Define constants, parameters, and boundary conditions of the solution
# =============================================================================

# Initial values of y and v
y0 = 0
v0 = 0.1 # Setting to 0 feels more accurate but breaks/results in a straight line

# phi = 0 @ y = 0 as inf. potential at y = 0 (boundary condition)
phi0 = 0 

# Max value of y, used in place of infinity
y1 = 6 # Tested for up to y1 = 14 but 6 looks best

# Denotes the number of iterations to be made in Runge-Kutta, and maximum
#   iterations in shooting method
N_rk = 1000
N_loops = 1000

# if phi < tol & phi < gradient at y = y1, phi is a ground state solution
tol = 1e-4

lam = -0.005 # Initial value of lambda (small and negative)
d_lam_0 = 0.1 # Default 'step size' in lambda


# =============================================================================
# Peform the shooting method; normalise and plot wavefunction
# =============================================================================

lam, y, phi = shooting_method(y0, v0, phi0, y1, N_rk, N_loops, lam, d_lam_0)
    
# Create figure
# Use ggplot for pretty graphs and create figure (and size)
plt.style.use('ggplot')
fig = plt.figure()
ax = plt.subplot(111)

# Find and fullscreen the plot
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Calculate the area under the wavefunction |phi|^2
mod_area = simps(abs(phi)**2, y)

# Normalise the wavefunction
phi = phi/np.sqrt(mod_area)

# Plot the ground state wavefunction
plot, = plt.plot(y, phi, color = 'blue', lw = 1.5)   

# =============================================================================
# Add titles, labels, potential boxes, and legend
# =============================================================================

# Generate plot title and add
title = ''.join([r'Normalised Ground State Wavefunction $\phi$ for Potential ',
          r'$V_0 = 10$ with Energy $\lambda =$', str(lam), r' (in terms of ',
          r'$\frac{\hbar}{2ma^2}$)','\n', r'and x-axis scaled by ',
          'Potential Width $a$'])
plt.title(title, fontsize = 22)

# Generate x and y labels and add
plt.xlabel(r'$\frac{x}{a}$', fontsize = 22)
plt.ylabel(r'$\phi$', fontsize = 22)

# Get the x and y limits
ylim = ax.get_ylim()
xlim = ax.get_xlim()

# Draw boxes to show the potentials
box_width_1 = 1
box_width_2 = xlim[1] - xlim[0]
box_height = ylim[1] - ylim[0]
box1 = patches.Rectangle(xy = (-1, ylim[0]), width = box_width_1,
                        height = box_height, color = 'black', 
                        alpha = 0.2)
box2 = patches.Rectangle(xy = (y0, ylim[0]), width = box_width_1,
                        height = box_height, color = 'green', 
                        alpha = 0.2)

box3 = patches.Rectangle(xy = (1, ylim[0]), width = box_width_2,
                        height = box_height, color = 'red', 
                        alpha = 0.2)
ax.add_patch(box1); ax.add_patch(box2); ax.add_patch(box3)

# Sneakily add a line to show the wavefunction is 0 for negative values of y
line = lines.Line2D([xlim[0], y0], [0, 0], color='blue', lw = 1.5, axes=ax)
ax.add_line(line)

# Add a legend
plt.legend((plot, box1, box2, box3), 
           ('2nd Order Runge-Kutta Solution', r'$V(\frac{x}{a}) = \infty$', 
            r'$V(\frac{x}{a}) = V_0$', r'$V(\frac{x}{a}) = 0$'),
           prop={'size': 12})