import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def circ(x,r,x0,y0):
    return y0 + np.sqrt(np.power(r,2) - np.power(x-x0,2))

def circ2(x,r,x0,y0):
    return y0 - np.sqrt(np.power(r,2) - np.power(x-x0,2))

def generate_random_arc_data(n_pts_gen=100, r_avg=2.0, x0_avg=5.0, y0_avg=5.0, rad_flucts=1e-3, max_deg=15.0, show=False):
	## Define the true parameters
	x0_true = x0_avg * 2.*(np.random.rand()-0.5)
	y0_true = y0_avg * 2.*(np.random.rand()-0.5)
	r_true  = r_avg * np.random.rand()

	## Select a random number of points forming an arc along the circle
	phi_min_rad = np.random.rand() * 2.*np.pi 
	phi_max_rad = (max_deg/360.0)* 2.*np.pi  + phi_min_rad
	rand_phis   = phi_min_rad + (phi_max_rad-phi_min_rad)*np.random.rand(int(n_pts_gen))
	rand_rads   = r_true + rad_flucts * (np.random.rand(int(n_pts_gen))-0.5)

	## Sort the array by angle
	sort_idxs = np.argsort(rand_phis)
	rand_phis = rand_phis[sort_idxs]
	rand_rads = rand_rads[sort_idxs]

	## Generate the x and y data points
	x_data = x0_true + rand_rads*np.cos(rand_phis) # avg_p_template.real
	y_data = y0_true + rand_rads*np.sin(rand_phis) # avg_p_template.imag

	if show:
		print("True radius:  ", r_true)
		print("True center: (", x0_true, "," , y0_true, ")")

	return x_data, y_data, (r_true, x0_true, y0_true)

def plot_sim_data(x_data, y_data, truth=None, fig_obj=None):
	if len(truth) is not 3:
		truth = None

	if fig_obj is None:
		fig_obj = plt.figure("Simulated Data and Truth")
	ax  = fig_obj.gca()

	ax.scatter(x_data,y_data,label="Data")
	ax.set_aspect('equal','box')

	if truth is not None:
		circ_xvals_true = np.linspace(start=truth[1]-truth[0], stop=truth[1]+truth[0], num=1000)
		circ_yvals_true = circ(circ_xvals_true,truth[0],truth[1],truth[2])
		circ_yvals_true_n = circ2(circ_xvals_true,truth[0],truth[1],truth[2])
		
		xlims = ax.get_xlim()
		ylims = ax.get_ylim()
		ax.plot(circ_xvals_true,circ_yvals_true,'k--', label="Truth")
		ax.plot(circ_xvals_true,circ_yvals_true_n,'k--')
		ax.set_xlim(xlims)
		ax.set_ylim(ylims)
	
	plt.legend(loc='best')
	return fig_obj

def estimate_circ_params(x_data, y_data, show=False):
	
	## Calculate a secant line
	slope = (y_data[-1] - y_data[0])/(x_data[-1] - x_data[0])
	incpt =  y_data[-1] - slope*x_data[-1]

	x_range = np.linspace(start=np.min(x_data),stop=np.max(x_data),num=1000)
	s_vals  = slope*x_range + incpt

	sec_len = np.sqrt( np.power(x_range[-1]-x_range[0],2) + np.power(s_vals[-1]-s_vals[0],2) )

	## Determine the middle of the secant line
	x_med   = np.median(x_range)
	y_med   = slope*x_med + incpt

	## Calculate a perpendicular bisector
	slope_pbs   = (-1./slope)
	incpt_pbs   = y_med - slope_pbs*x_med
	x_incpt_pbs = -incpt_pbs/slope_pbs

	## Find the intersection of the data arc and perpendicular bisector
	a_len = np.sqrt( np.power(np.min(x_data)-x_med,2) + np.power(np.min(y_data)-y_med,2) )
	b_len = np.min( np.sqrt( np.power(x_data-x_med,2) + np.power(y_data-y_med,2) ) )

	fa = 1 + np.power(slope_pbs,2)
	fb = -2*x_med + 2*slope_pbs*(incpt_pbs-y_med)
	fc = np.power(x_med,2) - np.power(b_len,2) + np.power(incpt_pbs-y_med,2)

	xstars = np.zeros(2)
	ystars = np.zeros(2)

	xstars[0] = (1./(2*fa)) * (-fb + np.sqrt(np.power(fb,2)-4*fa*fc))
	xstars[1] = (1./(2*fa)) * (-fb - np.sqrt(np.power(fb,2)-4*fa*fc))

	ystars    = slope_pbs*xstars + incpt_pbs

	## Choose which solution to use since the "star" point should be close to the median of the data
	dists = np.sqrt( np.power( xstars-np.median(x_data) , 2 ) + np.power( ystars-np.median(y_data) , 2) )

	xstar = xstars[ np.argmin( dists ) ]
	ystar = ystars[ np.argmin( dists ) ]

	## Use the other side of the arc
	idx1 = -1
	idx2 = -10

	arc_end_slope  = (y_data[idx1]-y_data[idx2])/(x_data[idx1]-x_data[idx2])
	arc_perp_slope = -1./arc_end_slope
	arc_end_incpt  = y_data[idx1] - arc_perp_slope*x_data[idx1]
	arc_x_incpt    = -arc_end_incpt/arc_perp_slope

	## Find where they cross
	x_cross = (arc_end_incpt-incpt_pbs) / (slope_pbs - arc_perp_slope)
	y_cross = arc_perp_slope*x_cross + arc_end_incpt

	## Now what's the distance from red V to black * 
	x0_est = x_cross
	y0_est = y_cross
	r_est  = np.sqrt( np.power(x0_est-xstar,2) + np.power(y0_est-ystar,2) )

	## Make a plot if requested
	if show:
		fig_obj = plt.figure("Parameter Estimation",figsize=(8,6))
		ax = fig_obj.gca()

		ax.plot(x_data,y_data,color='steelblue',marker='o',markersize=5,ls='-')
		ax.plot(x_range,s_vals,color='k',marker='.',markersize=1,ls='-')
		ax.scatter([x_med],[y_med],marker="*",color='y',zorder=100)
		ax.scatter([xstar],[ystar],marker="v",color='r',zorder=100)
		ax.scatter([x_data[idx1]],[y_data[idx1]],marker="^",color='r',zorder=100)
		ax.scatter([x_data[idx2]],[y_data[idx2]],marker="^",color='g',zorder=100)
		ax.scatter([x0_est],[y0_est],marker="*",color='k',zorder=100)

		ax.set_aspect('equal','box')

	return (r_est, x0_est, y0_est)

def fit_data_circle(x_data, y_data, est_vals):
	## Based on where the estimated center is relative to the data,
	## fit to a specific half of the circle
	check_above = np.median(y_data) > est_vals[2]

	if check_above: 
	    ## If your data is above your median (vertically), fit to the top half of the circle
	    popt, pcov = curve_fit(circ, x_data, y_data, p0=est_vals)
	else: 
	    ## If your data is below your median (vertically), fit to the bottom half of the circle
	    popt, pcov = curve_fit(circ2, x_data, y_data, p0=est_vals)

	return popt, pcov, check_above

def plot_data_est_fit(x_data, y_data, est_vals=None, fit_vals=None, truth=None, fig_obj=None):

	if fig_obj is None:
		fig_obj = plt.figure("Data vs Estimate vs Fit vs Truth",figsize=(8,6))
	ax = fig_obj.gca()

	## First plot the data 
	ax.plot(x_data,y_data,color='steelblue',marker='o',markersize=5,ls='-',label="Data")

	## Now plot the guess
	if est_vals is not None:
		if len(est_vals) == 3:
			x_range_ = np.linspace(start=est_vals[1]-est_vals[0],stop=est_vals[1]+est_vals[0],num=1000)

			y_vals_g  = circ( x_range_, est_vals[0], est_vals[1], est_vals[2])
			y_vals_g2 = circ2(x_range_, est_vals[0], est_vals[1], est_vals[2])

			ax.scatter([est_vals[1]],[est_vals[2]],marker="*",color='orange')
			ax.plot(x_range_, y_vals_g,  color='orange', ls='--', label="Guess")
			ax.plot(x_range_, y_vals_g2, color='orange', ls='--')

			print("Estimated values: r=", est_vals[0], "x0=", est_vals[1], "y0=", est_vals[2])

	## Now plot the fit result
	if fit_vals is not None:
		if len(fit_vals) == 3:
			x_range_ = np.linspace(start=fit_vals[1]-fit_vals[0],stop=fit_vals[1]+fit_vals[0],num=1000)

			y_vals_f  = circ( x_range_, fit_vals[0], fit_vals[1], fit_vals[2])
			y_vals_f2 = circ2(x_range_, fit_vals[0], fit_vals[1], fit_vals[2])

			ax.scatter([fit_vals[1]],[fit_vals[2]],marker="*",color='purple')
			ax.plot(x_range_, y_vals_f,  color='purple', ls='-',  label="Fit")
			ax.plot(x_range_, y_vals_f2, color='purple', ls='-')

			print("Optml fit values: r=", fit_vals[0], "x0=", fit_vals[1], "y0=", fit_vals[2])

	## Now plot the truth
	if truth is not None:
		if len(truth) == 3:
			x_range_ = np.linspace(start=truth[1]-truth[0],stop=truth[1]+truth[0],num=1000)

			y_vals_t  = circ( x_range_, truth[0], truth[1], truth[2])
			y_vals_t2 = circ2(x_range_, truth[0], truth[1], truth[2])

			ax.scatter([truth[1]],[truth[2]],marker="*",color='green')
			plt.plot(x_range_, y_vals_t,  color='green',  ls=':',  label="Truth")
			plt.plot(x_range_, y_vals_t2, color='green', ls=':')

			print("MC truth values:  r=", truth[0], "x0=", truth[1], "y0=", truth[2])
			if fit_vals is not None:
				if len(fit_vals) == 3:
					print("Error percent:    r=", 100.0*(fit_vals[0]-truth[0])/truth[0], 
						                   "x0=", 100.0*(fit_vals[1]-truth[1])/truth[1], 
						                   "x0=", 100.0*(fit_vals[2]-truth[2])/truth[2])
	
	
	plt.legend(loc='lower right')
	ax.set_aspect('equal','box')

	return fig_obj

def plot_fit_residuals(x_data, y_data, fit_vals, check_above):
	residuals = y_data - ( circ(x_data, fit_vals[0], fit_vals[1], fit_vals[2]) if check_above else circ2(x_data, fit_vals[0], fit_vals[1], fit_vals[2]) )
	f_resdls  = residuals/y_data

	plt.figure("Residual Histogram")
	plt.title("Distribution of fit residuals")
	plt.xlabe("Residuals [%]")
	plt.hist(100.0*f_resdls,bins=50)

if __name__ == "__main__":
	x_data, y_data, truth = generate_random_arc_data(show=True)

	plot_sim_data(x_data, y_data, truth=truth)

	par_est = estimate_circ_params(x_data, y_data, show=True)

	par_fit, pcov_fit, circ_ch = fit_data_circle(x_data, y_data, par_est)

	plot_data_est_fit(x_data, y_data, est_vals=par_est, fit_vals=par_fit, truth=truth)

	plot_fit_residuals(x_data, y_data, par_fit, circ_ch)

	plt.show()
