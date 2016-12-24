import numpy as np
from fair_main import *

def main():

  #Import the RCP8.5 forcing and emissions
  emms_file = 'RCPs/8.5/RCP85_EMISSIONS.csv'
  rcp_co2_e, rcp_ch4_e, rcp_n2o_e, years = load_rcp_emissions(emms_file)
  rf_file = 'RCPs/8.5/RCP85_MIDYEAR_RADFORCING.csv'
  rcp_tot_rf, rcp_ant_rf, rcp_co2_rf, rcp_ch4_rf, rcp_ch4_rf, rcp_ghg_rf,rcp_ky_rf = load_rcp_forcing(rf_file)

  #Default FAIR carbon cycle parameters
  r0 = 35.0   #r_{0} in ACP paper
  rc = 0.02              #r_{C} in ACP paper
  rt = 4.5             #r_{T} in ACP paper
  pre_indust_co2 = 278.0

  #Integrate model to get output concs and temps. Emissions are required. All other model parameter values are set at run time and can be changed (e.g. TCR).
  temps, concs = fair_scm(rcp_co2_e,other_rf=rcp_tot_rf-rcp_co2_rf,TCR=1.75,ECS=3.0,rt=rt,r0=r0,rc = rc,a=3.74/np.log(2.0),pre_indust_co2=pre_indust_co2)

  #Plot the temperatures against time
  plt.plot(years,temps)
  plt.show()

  return

if __name__ == "__main__":
  main()