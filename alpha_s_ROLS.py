import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit, root, brentq
import matplotlib.pyplot as plt

#Definition of the three models
def rp_p(z,p):
    return 1/(1 + 1/z**p)

def rp_r(z,p):
    return 1/(1+1/z)**p

def rp_l(z,p):
    return (1 - np.exp(-z))**p

def b0(n):
    return 11 - 2/3*n

def K0(n):
    return b0(n) / (4 * np.pi)

def rho0_p(z, n, p):
    return K0(n) * (-np.pi) * rp_p(z, p)

def rho0_r(z, n, p):
    return K0(n) * (-np.pi) * rp_r(z, p)

def rho0_l(z, n, p):
    return K0(n) * (-np.pi) * rp_l(z, p)

def e0_p(z, n, p):
    def integrand_p(x, z, n, p):
        return rho0_p(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_p, 0, np.inf, args=(z, n, p))[0]

def e0_r(z, n, p):
    def integrand_r(x, z, n, p):
        return rho0_r(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_r, 0, np.inf, args=(z, n, p))[0]

def e0_l(z, n, p):
    def integrand_l(x, z, n, p):
        return rho0_l(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_l, 0, np.inf, args=(z, n, p))[0]

def epsilon0_p(t, L, n, p):
    return e0_p(t / L**2, n, p) - e0_p(0, n, p)

def epsilon0_r(t, L, n, p):
    return e0_r(t / L**2, n, p) - e0_r(0, n, p)

def epsilon0_l(t, L, n, p):
    return e0_l(t / L**2, n, p) - e0_l(0, n, p)

def b1(n):
    return 102 - 38/3*n

def K1(n):
    return b1(n) / (4 * np.pi * b0(n))

def F(x):
    if 0 < x < 1:
        return -np.pi - np.arctan(np.pi / np.log(x))
    elif x==1:
        return -np.pi/2
    else:
        return -np.arctan(np.pi / np.log(x))
    
def rho1_p(z, n, p):
    return K1(n) * rp_p(z, p) * F(z)

def rho1_r(z, n, p):
    return K1(n) * rp_r(z, p) * F(z)

def rho1_l(z, n, p):
    return K1(n) * rp_l(z, p) * F(z)

def e1_p(z, n, p):
    def integrand_p(x, z, n, p):
        return rho1_p(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_p, 0, np.inf, args=(z, n, p))[0]

def e1_r(z, n, p):
    def integrand_r(x, z, n, p):
        return rho1_r(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_r, 0, np.inf, args=(z, n, p))[0]

def e1_l(z, n, p):
    def integrand_l(x, z, n, p):
        return rho1_l(x, n, p) / ((x + z) * (x + 1))
    
    return (1 - z) / np.pi * quad(integrand_l, 0, np.inf, args=(z, n, p))[0]
    
def epsilon1_p(t, L, n, p):
    return e1_p(t / L**2, n, p)- e1_p(0, n, p)

def epsilon1_r(t, L, n, p):
    return e1_r(t / L**2, n, p)- e1_r(0, n, p)

def epsilon1_l(t, L, n, p):
    return e1_l(t / L**2, n, p)- e1_l(0, n, p)

def epsilon_p(t, L, n, p):
    return epsilon0_p(t, L, n, p) + epsilon1_p(t, L, n, p)

def epsilon_r(t, L, n, p):
    return epsilon0_r(t, L, n, p) + epsilon1_r(t, L, n, p)

def epsilon_l(t, L, n, p):
    return epsilon0_l(t, L, n, p) + epsilon1_l(t, L, n, p)

def alpha_p(t, L, n, p):
    return 1/epsilon_p(t, L, n, p)

def alpha_r(t, L, n, p):
    return 1/epsilon_r(t, L, n, p)

def alpha_l(t, L, n, p):
    return 1/epsilon_l(t, L, n, p)

'''
#Plot of regularizing function
x=np.linspace(0.0,0.6,600)
plt.plot(x,rp_p(x/0.3,0.8), color='b', label=r'$[\hat{r}_p](\sigma)$')
plt.plot(x,rp_r(x/0.3,0.8), color='r', label=r'$[\breve{r}_p](\sigma)$')
plt.plot(x,rp_l(x/0.3,0.8), color='g', label=r'$[\tilde{r}_p](\sigma)$')

plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.3))


plt.xlabel(r'$\sigma$ [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$r_p(\sigma)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

plt.xlim(0,0.55)
plt.ylim(0,0.9)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)

plt.savefig('/Users/rocco/Desktop/alpha_s py/rp123_py.png', dpi=500)
plt.show()


#Plot of epsilon
x=np.linspace(0.0,0.6,600)
vepsilon_p = np.vectorize(epsilon_p, excluded=set([1,2,3]))
vepsilon_r = np.vectorize(epsilon_r, excluded=set([1,2,3]))
vepsilon_l = np.vectorize(epsilon_l, excluded=set([1,2,3]))

plt.plot(x,vepsilon_p(x,0.3,5,0.8), color='b', label=r'$[\hat{\overline{\varepsilon}}_s]_{an}(t)$')
plt.plot(x,vepsilon_r(x,0.3,5,0.8), color='r', label=r'$[\breve{\overline{\varepsilon}}_s]_{an}(t)$')
plt.plot(x,vepsilon_l(x,0.3,5,0.8), color='g', label=r'$[\tilde{\overline{\varepsilon}}_s]_{an}(t)$')

plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.3))


plt.xlabel(r't [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$[\overline{\varepsilon}_s]_{an}(t)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

plt.xlim(0,0.55)
plt.ylim(0,2.3)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)

plt.savefig('/Users/rocco/Desktop/alpha_s py/epsilon123_py.png', dpi=500)
plt.show()


#Plot of alpha
x=np.linspace(0.01,0.6,600)
valpha_p = np.vectorize(alpha_p, excluded=set([1,2,3]))
valpha_r = np.vectorize(alpha_r, excluded=set([1,2,3]))
valpha_l = np.vectorize(alpha_l, excluded=set([1,2,3]))

plt.plot(x,valpha_p(x,0.3,5,0.8), color='b', label=r'$[\hat{\overline{\alpha}}_s]_{an}(t)$')
plt.plot(x,valpha_r(x,0.3,5,0.8), color='r', label=r'$[\breve{\overline{\alpha}}_s]_{an}(t)$')
plt.plot(x,valpha_l(x,0.3,5,0.8), color='g', label=r'$[\tilde{\overline{\alpha}}_s]_{an}(t)$')

plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.6))


plt.xlabel(r't [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$[\overline{\alpha}_s]_{an}(t)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

plt.xlim(0,0.55)
plt.ylim(0,2.3)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)

plt.savefig('/Users/rocco/Desktop/alpha_s py/alpha123_py.png', dpi=500)
plt.show()
'''

#Fit with the experimental data

L=0.3

def alpha_p_5(t, p):
    return alpha_p(t, L, 5, p)

def alpha_r_5(t, p):
    return alpha_r(t, L, 5, p)

def alpha_l_5(t, p):
    return alpha_l(t, L, 5, p)

#Data
tJ=[14**2, 22**2, 34.6**2, 38.3**2, 35**2, 43.8**2]
JADE=[0.1536, 0.1407, 0.1346, 0.1355, 0.1391, 0.1289]
JADEerr=[np.sqrt(32**2 + 64**2 + 28**2 + 74**2)/10**4, 
np.sqrt(28**2 + 34**2 + 21**2 + 24**2)/10**4, 
np.sqrt(7**2 + 19**2 + 31**2 + 11**2)/10**4, 
np.sqrt(21**2 + 42**2 + 38**2 + 20**2)/10**4, 
np.sqrt(6**2 + 17**2 + 33**2 + 12**2)/10**4, 
np.sqrt(12**2 + 11**2 + 38**2 + 19**2)/10**4]

tL=[133**2,161**2,172**2]
LEP2=[0.113, 0.105, 0.103]
LEP2err=[np.sqrt(3^2 + 7^2)/10**3, np.sqrt(3^2 + 6^2)/10**3, np.sqrt(3^2 + 6^2)/10**3]

tC=[474**2, 664**2, 896**2]
CMS=[0.0936, 0.0894, 0.0889]
CMSerr=[41/10**4, 31/10**4, 34/10**4]

valpha_p_5 = np.vectorize(alpha_p_5, excluded=set([1]))
valpha_r_5 = np.vectorize(alpha_r_5, excluded=set([1]))
valpha_l_5 = np.vectorize(alpha_l_5, excluded=set([1]))

t=tJ+tL
tarr=np.array(t)
y=JADE+LEP2
err_y=JADEerr+LEP2err

# Execution of the fit
bounds = ([0], [1])
p0=[0.7]
params_p, covariance_p = curve_fit(valpha_p_5, t, y, bounds=bounds, p0=p0, sigma=err_y)
params_r, covariance_r = curve_fit(valpha_r_5, t, y, bounds=bounds, p0=p0, sigma=err_y)
params_l, covariance_l = curve_fit(valpha_l_5, t, y, bounds=bounds, p0=p0, sigma=err_y)

# Extract and print parameters
p_fit_p = params_p
p_fit_r = params_r
p_fit_l = params_l
sigma_fit_p = np.sqrt(np.diag(covariance_p))
sigma_fit_r = np.sqrt(np.diag(covariance_r))
sigma_fit_l = np.sqrt(np.diag(covariance_l))
print("p1:",p_fit_p," con incertezza:", sigma_fit_p[0])
print("p2:",p_fit_r," con incertezza:", sigma_fit_r[0])
print("p3:",p_fit_l," con incertezza:", sigma_fit_l[0])

# Definition of reduced chi squared
def reduced_chi_square(y_observed, y_predicted, sigma, num_parameters):
    residuals = y_observed - y_predicted
    dof = len(y_observed) - num_parameters  # Gradi di libertà
    chi_square = np.sum((residuals / sigma) ** 2)
    reduced_chi_square = chi_square / dof  # Chi quadro ridotto
    return reduced_chi_square

# Calculation of predicted values
y_pred_p = valpha_p_5(t, *params_p)
y_pred_r = valpha_r_5(t, *params_r)
y_pred_l = valpha_l_5(t, *params_l)

# Calculation of reduced chi squared
num_parameters = len(params_p)  # Numero dei parametri stimati
chi_squared_reduced_p = reduced_chi_square(y, y_pred_p, err_y, num_parameters)
chi_squared_reduced_r = reduced_chi_square(y, y_pred_r, err_y, num_parameters)
chi_squared_reduced_l = reduced_chi_square(y, y_pred_l, err_y, num_parameters)

print("Chi quadro ridotto 1:", chi_squared_reduced_p)
print("Chi quadro ridotto 2:", chi_squared_reduced_r)
print("Chi quadro ridotto 3:", chi_squared_reduced_l)


#Plots with the fitted curves
m=np.floor(np.min(tarr))-10
M=np.ceil(np.max(tarr))+10

#First parametrization
x=np.linspace(m,M,3000)
plt.errorbar(t, y, yerr=err_y, fmt='o', markersize=5, capsize=5)
plt.plot(x,valpha_p_5(x,p_fit_p), color='b')
plt.xlabel(r't [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$[\hat{\overline{\alpha}}_s]_{an}(t)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)
#plt.savefig('/Users/rocco/Desktop/alpha_s py/fit_p.png', dpi=500)
plt.show()

#Second parametrization
x=np.linspace(m,M,3000)
plt.errorbar(t, y, yerr=err_y, fmt='o', markersize=5, capsize=5)
plt.plot(x,valpha_r_5(x,p_fit_r), color='r')
plt.xlabel(r't [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$[\breve{\overline{\alpha}}_s]_{an}(t)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)
#plt.savefig('/Users/rocco/Desktop/alpha_s py/fit_r.png', dpi=500)
plt.show()

#Third parametrization
x=np.linspace(m,M,3000)
plt.errorbar(t, y, yerr=err_y, fmt='o', markersize=5, capsize=5)
plt.plot(x,valpha_l_5(x,p_fit_l), color='g')
plt.xlabel(r't [GeV$^2$]')
ax = plt.gca()
ax.set_ylabel(r'$[\tilde{\overline{\alpha}}_s]_{an}(t)$', rotation=0, labelpad=20)
ax.yaxis.set_label_coords(-0.1, 1.02)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)
#plt.savefig('/Users/rocco/Desktop/alpha_s py/fit_l.png', dpi=500)
plt.show()



#Calculation of p corresponding to alpha_s(MZ**2)=0.1180

def alpha_p_root(p):
    return alpha_p(91.1**2,0.3,5,p)-0.1180

def alpha_r_root(p):
    return alpha_r(91.1**2,0.3,5,p)-0.1180

def alpha_l_root(p):
    return alpha_l(91.1**2,0.3,5,p)-0.1180

#print(brentq(alpha_p_root,0,1))
#print(brentq(alpha_r_root,0,1))
#print(brentq(alpha_l_root,0,1))


#print(G_l(2,3))
#print(rhoa_l(100,3))
#print(A_l(0.001, 0.3, 3))
