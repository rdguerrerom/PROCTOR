import PROCTOR
import numpy as np
n_points = 1024
x = np.linspace(0.0,10.0, num=n_points, dtype=np.double)
y1=x**2
y2=x**3
psi = np.linspace(0.0,10.0, num=n_points, dtype=np.complex128)
approximant_order = 6
sim_test = PROCTOR.Simulation1D(x,0.20408163, 0.1, n_points, 1, 1, 1, approximant_order)
sim_test.set_wave_packet(0,"Singlet",psi)
sim_test.set_PES(0, "Singlet", y1)
sim_test.set_PES(1, "Singlet", y2)
sim_test.set_DM(0, "Singlet", y1)
sim_test.set_DM(1, "Singlet", y2)
sim_test.set_NACME(0, 1, "Singlet", y1)
sim_test.set_NACME(1, 0, "Singlet", y2)
sim_test.set_TDM(0, 1, "Singlet", y1)
sim_test.set_TDM(1, 0, "Singlet", y2)
sim_test.set_SOC(0, 1, "Singlet","Doublet", y1)
sim_test.set_SOC(1, 0, "Singlet","Doublet", y2)
sampling_test = sim_test.get_perturbation_sampling_forward()
print( sampling_test  )
sim_test.set_perturbation_signal_samples(sampling_test)
for i in range(1000):
    sim_test.step_forward()
#sim_test.symmetrized_apprroximant(0.01, 1.0)
# => getters <= #
#print( sim_test.get_wave_packet(0,"Singlet",psi) )
#print( sim_test.get_DM(0, "Singlet", y1) )
#print( sim_test.get_DM(1, "Singlet", y2) )
#print( sim_test.get_NACME(0, 1, "Singlet", y1) )
#print( sim_test.get_NACME(1, 0, "Singlet", y2) )
#print( sim_test.get_TDM(0, 1, "Singlet", y1) )
#print( sim_test.get_TDM(1, 0, "Singlet", y2) )
#print( sim_test.get_SOC(0, 1, "Singlet","Doublet", y1) )
#print( sim_test.get_SOC(1, 0, "Singlet","Doublet", y2) )

