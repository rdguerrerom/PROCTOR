import PROCTOR
import numpy as np
x = np.linspace(0.0,10.0, num=50, dtype=np.double)
y1=x**2
y2=x**3
psi = np.linspace(0.0,10.0, num=50, dtype=np.complex128)
sim_test = PROCTOR.Simulation1D(x,0.20408163, 0.1, 50, 1, 0, 0)
sim_test.set_wave_packet(0,"Singlet",psi)
sim_test.set_DM(0, "Singlet", y1)
sim_test.set_DM(1, "Singlet", y2)
sim_test.set_NACME(0, 1, "Singlet", y1)
sim_test.set_NACME(1, 0, "Singlet", y2)
sim_test.set_TDM(0, 1, "Singlet", y1)
sim_test.set_TDM(1, 0, "Singlet", y2)
sim_test.set_SOC(0, 1, "Singlet","Doublet", y1)
sim_test.set_SOC(1, 0, "Singlet","Doublet", y2)
