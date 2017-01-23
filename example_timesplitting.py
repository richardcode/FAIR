from fair_main_rev import *
import matplotlib.pyplot as plt

def controller(k,temp_current,temp_goal,i=0,d=0):

  sf = k*temp_goal - temp_current

  return sf


def fair_scm_control(base_emissions,temp_goal,k,start_state):

  temps_out = []
  sf_out = []

  for x in range(0,emissions.shape[0]):

    if x == 0:
      [concs_step, temps_step], restart = fair_scm(emissions[x:(x+1)],restart_out=True,restart_in=start_state)
      temps_out.append(temps_step[-1])
    else:
    
      #Controller in here
      sf =  controller(k,temps_step[-1],temp_goal)
      sf_out.append(sf)
      
      [concs_step, temps_step], restart = fair_scm(sf*emissions[(x-1):(x+1)],restart_out=True,restart_in=restart)
    temps_out.append(temps_step[-1])

  return temps_out, sf_out


#Load in RCP8.5 to be 'base_emissions' and integrate up to 2015 to get 'start_start'

ks = np.arange(0,5,step=0.1)


temp_goal = 3.0


temps_out_all = []

for k in ks:

  temps_out, sf_out = fair_scm_control(base_emissions,temp_goal,k,start_state)
  temps_out_all.append(temps_out)







#OLD CODE BENEATH

#Create emissions array
emissions = np.zeros(shape=(100,))
emissions[0:10] = np.arange(0,10)
emissions[10:] = 10.0

#Integrate the first half of the emissions array and get the restart dump
[concs_first, temps_first], restart = fair_scm(emissions[:50],restart_out=True)

#Integrate the first half of the emissions array and get the restart dump
[concs_second, temps_second] = fair_scm(emissions[(50-1):],restart_out=False,restart_in=restart)

#First entry of second array is the last point in first array (i.e. it is duplicated)
print concs_first.shape[0] + concs_second.shape[0]
print emissions.shape[0]

#Correct way to combine them
concs_comb = np.zeros_like(emissions)
concs_comb[:50] = concs_first
concs_comb[50:] = concs_second[1:]

#Integrate the everything at once
[concs_all, temps_all] = fair_scm(emissions,restart_out=False)

#Results are identical to numerical error...
plt.plot(concs_all)
plt.plot(concs_comb)
plt.show()


temp_goal = 4.0


