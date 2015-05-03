#####################################################################
## 
#####################################################################

import os, time, sys
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants
from lmfit import minimize, Parameters, report_fit

start = time.clock()
print 'Start the timer...'

# Define some useful constants first:
c = constants.c.cgs.value # Speed of light (cm/s)
k_B = constants.k_B.cgs.value # Boltzmann coefficient (erg/K)
h = constants.h.cgs.value # Planck constant (erg*s)

def __readascii__(infile):
	"""
	Read in an ASCII file, first column is velocity/frequency axis,
	second column is intensity/brightness temperature.
	Return two numpy arrays.
	"""
	temp = open(infile, 'r')
	text = temp.readlines()
	spec = np.empty(len(text))
	vaxis = np.empty(len(text))
	for line in range(len(text)):
		vaxis[line] = np.float(text[line].split()[0])
		spec[line] = np.float(text[line].split()[1])
	temp.close()
	del temp

	return spec, vaxis

def __h2co_init__():
	"""
	Initialize the parameters. No input keywords.
	"""
	h2co_info = {}
	h2co_info['Km1'] = [0, 2, 2] # K_{-1}
	h2co_info['E'] = [10.4834, 57.6086, 57.6120] # Upper level energy (K)
	h2co_info['frest'] = [218.22219, 218.47563, 218.76007] # Rest frequencies (GHz)
	h2co_info['gk'] = [1./4, 1./4, 1./4] # Degeneracy = g_K*g_nuclear
	h2co_info['mu'] = 2.3317e-18 # Permanent dipole moment (esu*cm)
	h2co_info['A'] = 281.97056 # Rotational constant (GHz)
	h2co_info['B'] = 38.83399 # Another rotational constant (GHz)
	h2co_info['C'] = 34.00424 # Another rotational constant (GHz)

	return h2co_info

def __gauss_tau__(axis,p):
	"""
	Genenerate a Gaussian model given an axis and a set of parameters.
	p: [T, Ntot, fsky, sigma, Lid]
	"""
	T= p[0]; Ntot = p[1]; fsky = p[2]; sigma = p[3]; Lid = p[4]
	K = h2co_info['Km1'][Lid]

	phijk = 1/sqrt(2 * pi) / (sigma * 1e9) * np.exp(-0.5 * (axis - fsky)**2 / sigma**2)
	Ajk = (64 * pi**4 * (h2co_info['frest'][Lid] * 1e9)**3 * h2co_info['mu']**2 / 3 / h / c**3) * (J**2 - K**2) / (J * (2*J + 1))
	gjk = (2*J + 1) * h2co_info['gk'][K]
	#Q = 3.89 * T**1.5 / (-1.0 * expm1(-524.8 / T))**2
	Q = 168.7 * T**1.5 / sqrt(h2co_info['A'] * h2co_info['B'] * h2co_info['C'])
	Njk = Ntot * (gjk / Q) * exp(-1.0 * h2co_info['E'][K] / T)

	tau = (h * c**2 * Njk * Ajk) / (8 * pi * h2co_info['frest'][K] * 1e9 * k_B * T) * phijk
	f = T * (1 - np.exp(-1.0 * tau))	

	return f

def __model_11__(params, faxis, spec):
	"""
	Model hyperfine components of CH3CN.
	Then subtract data.
	params: [T, Ntot, fsky, sigma]
	"""
	T = params['T'].value
	Ntot = params['Ntot'].value
	fsky = params['fsky'].value
	sigma = params['sigma'].value

	fsky_k = h2co_info['frest'] + (fsky - h2co_info['frest'][0])		

	model = np.zeros(len(faxis))
	for Lid in arange(3):
		model += __gauss_tau__(faxis, [T, Ntot, fsky_k[Lid], sigma, Lid])

	return model - spec

clickvalue = []
def onclick(event):
	print 'The frequency you select: %f' % event.xdata
	clickvalue.append(event.xdata)

def fit_spec(spec, faxis, Jupp=3, cutoff=0.009, varyf=2, interactive=True, mode='single'):
	"""
	Fit the hyperfine lines of CH3CN, derive best-fitted Trot and Ntot.
	Input:
		spec: the spectra
		faxis: the frequency axis
		Jupp: J of the upper level
		cutoff: not used...
		varyf: number of channels to vary after you select the Vlsr
		interactive: true or false
		mode: single or double, components along the line of sight (to be done...)
	"""
	# Define the J, K numbers as global variables:
	# (Not recommended by Python experts...)
	global J
	J = Jupp

	if interactive:
		plt.ion()
		f = plt.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.ion()
			plt.plot(faxis, spec, 'k-', label='Spectrum')
			cutoff_line = [cutoff] * len(faxis)
			cutoff_line_minus = [-1.0*cutoff] * len(faxis)
			plt.plot(faxis, cutoff_line, 'r-')
			plt.plot(faxis, cutoff_line_minus, 'r-')
			plt.xlabel(r'Sky Frequency (GHz)', fontsize=20, labelpad=10)
			plt.ylabel(r'$T_{\nu}$ (K)', fontsize=20)
			plt.text(0.02, 0.92, sourcename, transform=ax.transAxes, color='r', fontsize=15)
			#plt.ylim([-10,60])
			#clickvalue = []
			if mode == 'single':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select a Vlsr...')
				#print clickvalue
				if len(clickvalue) >= 1:
					print 'Please select at least one velocity! The last one will be used.'
					vlsr1 = clickvalue[-1]
					vlsr1 = c * (1 - vlsr1/h2co_info['frest'][0]) / 1e5
				elif len(clickvalue) == 0:
					vlsr1 = 0.0
				print 'Or input one velocity manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 1:
					vlsr1 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The Vlsr is %0.2f km/s' % vlsr1
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
				vlsr2 = 0.0
			elif mode == 'double':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsrs...')
				print clickvalue
				if len(clickvalue) >= 2:
					print 'Please select at least two velocities! The last two will be used.'
					vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
				elif len(clickvalue) == 1:
					vlsr1 = clickvalue[-1]
					vlsr2 = 0.0
				elif len(clickvalue) == 0:
					vlsr1,vlsr2 = 0.0,0.0
				print 'Or input two velocities manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 2:
					vlsr1,vlsr2 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The two Vlsrs are %0.2f km/s and %0.2f km/s.' % (vlsr1,vlsr2)
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
			else:
				vlsr1,vlsr2 = 0.0,0.0
		else:
			if mode == 'single':
				if spec_low.max() >= cutoff:
					print 'Reserved space...'
				else:
					vlsr1 = 0.0
				vlsr2 = 0.0
			elif mode == 'double':
				vlsr1,vlsr2 = 86.0,88.0
			else:
				vlsr1,vlsr2 = 0.0,0.0

		plt.text(0.02, 0.85, r'$V_\mathrm{lsr}$=%.1f km/s' % vlsr1, transform=ax.transAxes, color='r', fontsize=15)
		fsky_init = h2co_info['frest'][0] * (1 - vlsr1 * 1e5 / c)

        # Add 4 parameters:
		params = Parameters()
		if vlsr1 != 0:
			params.add('Ntot', value=1e15, min=0, max=1e25)
			params.add('T', value=100, min=10)
			params.add('sigma', value=0.00156, min=0, max=0.050)
			#params.add('sigma', value=0.0027, min=0, max=0.050)
			if varyf > 0:
				params.add('fsky', value=fsky_init, min=fsky_init-varyf*chanwidth, \
				max=fsky0_init+varyf*chanwidth)
			elif varyf == 0:
				params.add('fsky', value=fsky_init, vary=False)
		if vlsr2 != 0:
			print 'Reserved for two-component fitting.'
		
		# Run the non-linear minimization:
		if vlsr1 != 0 and vlsr2 != 0:
			result = minimize(__model_11_2c__, params, args=(faxis, spec))
		elif vlsr1 != 0 or vlsr2 != 0:
			result = minimize(__model_11__, params, args=(faxis, spec))
		else:
			unsatisfied = False
			continue

		final = spec + result.residual
		#report_fit(params)

		if interactive:
			plt.plot(faxis, final, 'r', label='Best-fitted model')
			if vlsr1 != 0 and vlsr2 != 0:
				print 'Reserved for two-component fitting.'
			elif vlsr1 != 0 or vlsr2 != 0:
				plt.text(0.02, 0.80, r'T$_{rot}$=%.1f($\pm$%.1f) K' % (params['T'].value,params['T'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.75, r'N$_{tot}$=%.2e($\pm$%.2e) cm$^{-2}$' % (params['Ntot'].value,params['Ntot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
				plt.text(0.02, 0.70, r'FWHM=%.2f($\pm$%.2f) km/s' % (c*params['sigma'].value/h2co_info['frest'][0]/1e5*2.355,c*params['sigma'].stderr/h2co_info['frest'][0]/1e5*2.355), transform=ax.transAxes, color='r', fontsize=15)
			plt.legend()
			plt.show()
			print 'Is the fitting ok? y/n'
			yn = raw_input()
			if yn == 'y':
				unsatisfied = False
				currentT = time.strftime("%Y-%m-%d_%H:%M:%S")
				plt.savefig('H2CO_fitting_'+currentT+'.png')
			else:
				unsatisfied = True
			#raw_input('Press any key to continue...')
			f.clear()
		else:
			unsatisfied = False

###############################################################################

h2co_info = __h2co_init__()

# Read the ASCII file.
# The frequency axis is assumed to be in GHz, and the y axis is T_B in K.
spec, faxis = __readascii__('core2p1.txt')
sourcename = 'mm2p1'
chanwidth = abs(faxis[0] - faxis[-1]) / len(faxis)
print 'Channel width is %.4f GHz' % chanwidth
print 'Channel number is %d' % len(faxis)

# Convert from Jy/beam to K:
spec = 1.224e6 * spec / (h2co_info['frest'][0])**2 / (1.06 * 0.76)

# Reverse the axis:
#spec = spec[::-1]
#faxis = faxis[::-1]

# Run the fitting:
fit_spec(spec, faxis, Jupp=3, cutoff=0.1, varyf=0, interactive=True, mode='single')


elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)

