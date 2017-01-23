import numpy as np
from scipy.optimize import *
from scipy.interpolate import interp1d

############################################
# FAIR SIMPLE CLIMATE MODEL - WITH CARBON CYCLE
#
# written by Richard Millar (millar@atm.ox.ac.uk)
############################################

def load_rcp_emissions(emms_file):

  #Loads a CO2 emissions timeseries
  dt = np.dtype({'names':["YEARS","FossilCO2","OtherCO2","CH4","N2O","SOx","CO","NMVOC","NOx","BC","OC","NH3","CF4","C2F6","C6F14","HFC23","HFC32","HFC43_10","HFC125","HFC134a","HFC143a","HFC227ea","HFC245fa","SF6","CFC_11","CFC_12","CFC_113","CFC_114","CFC_115","CARB_TET","MCF","HCFC_22","HCFC_141B","HCFC_142B","HALON1211","HALON1202","HALON1301","HALON2402","CH3BR","CH3CL"],'formats':40*["f8"]})
  emm_data = np.genfromtxt(emms_file,skip_header=38,delimiter=',',dtype=dt)

  return emm_data['FossilCO2'] + emm_data['OtherCO2'] , emm_data['CH4'], emm_data['N2O'], emm_data['YEARS']

def load_rcp_forcing(forc_file):

  #Loads a CO2 emissions timeseries
  dt = np.dtype({'names':["YEARS","TOTAL_INCLVOLCANIC_RF","VOLCANIC_ANNUAL_RF","SOLAR_RF","TOTAL_ANTHRO_RF","GHG_RF","KYOTOGHG_RF","CO2CH4N2O_RF","CO2_RF","CH4_RF","N2O_RF","FGASSUM_RF","MHALOSUM_RF","CF4","C2F6","C6F14","HFC23","HFC32","HFC43_10","HFC125","HFC134a","HFC143a","HFC227ea","HFC245fa","SF6","CFC_11","CFC_12","CFC_113","CFC_114","CFC_115","CARB_TET","MCF","HCFC_22","HCFC_141B","HCFC_142B","HALON1211","HALON1202","HALON1301","HALON2402","CH3BR","CH3CL","TOTAER_DIR_RF","OCI_RF","BCI_RF","SOXI_RF","NOXI_RF","BIOMASSAER_RF","MINERALDUST_RF","CLOUD_TOT_RF","STRATOZ_RF","TROPOZ_RF","CH4OXSTRATH2O_RF","LANDUSE_RF","BCSNOW_RF"],'formats':54*["f8"]})
  forc_data = np.genfromtxt(forc_file,skip_header=59,delimiter=',',dtype=dt)

  return forc_data['TOTAL_INCLVOLCANIC_RF'], forc_data['TOTAL_ANTHRO_RF'], forc_data['CO2_RF'], forc_data['CH4_RF'], forc_data['N2O_RF'], forc_data['GHG_RF'], forc_data['KYOTOGHG_RF']

def load_rcp_concentrations(conc_file):

  data_concs = np.genfromtxt(conc_file,skip_header=39,delimiter=',')
  concs_years = data_concs[:,0]
  n2o_cons = data_concs[:,5]

  return concs_years, n2o_cons

def inversion(TCR=1.5,ECS=2.5,d1=4.1,d2=239.0,a=5.35):

  #Invert the thermal impulse response function to derive coefficents consistent with
  #specific TCR and ECS 

  ecstcr = np.array([ECS,TCR])
  for_mat = np.empty([2,2])

  for_mat[0,:] = np.array([1.0, 1.0])
  for_mat[1,:] = [1.0 - (d1/70.0)*(1.0- np.exp(-70.0/d1)),1.0 - (d2/70.0)*(1.0- np.exp(-70.0/d2))]
  for_mat= np.matrix(a*np.log(2.0)*for_mat)

  inverse = np.linalg.inv(for_mat)
  se = inverse*(np.matrix(ecstcr).T)
  [c1,c2]=np.array(se)
  c1=c1[0]
  c2=c2[0]

  return c1,c2


def integral_2_carbon_tempint(out_concs,out_temp,t,emissions,other_rf,c1,c2,d1=4.1,d2=239.0,a0=0.2173,a1=0.2240,a2=0.2824,a3=0.2763,b0=1000000,b1=394.4,b2=36.54,b3=4.304,t_iirf=100.0,rt=4.5,temp_off=-0.5,r0=35.0,rc = 0.02,a=5.35,pre_indust_co2=278.0,c=2.123,iirf_max=97.0,restart_in=False,restart_out=False):


    #Do the integral in terms of the individual response boxes
    out_concs[:] = 0.0
    C_0 = np.zeros_like(out_concs)
    C_1 = np.zeros_like(out_concs)
    C_2 = np.zeros_like(out_concs)
    C_3 = np.zeros_like(out_concs)
    
    
    radiative_forcing = np.zeros_like(out_concs)
    cumuptake = np.zeros_like(emissions)
    
    #Initialise the temperature anomaly boxes
    out_temp[:] = 0.0
    T_1 = np.zeros_like(out_temp)
    T_2 = np.zeros_like(out_temp)

    
    if restart_in:
      C_0[0]=restart_in[0]
      C_1[0]=restart_in[1]
      C_2[0]=restart_in[2]
      C_3[0]=restart_in[3]
      T_1[0]=restart_in[4]
      T_2[0]=restart_in[5]
      cumuptake[0] = restart_in[6]
        
      out_concs[0]=C_0[0]+C_1[0]+C_2[0]+C_3[0]
      out_temp[0]=T_1[0]+T_2[0]
    else:              

      #Initialise the carbon pools to be correct for first timestep in numerical method
      C_0[0] = emissions[0] / c * 0.5 * a0
      C_1[0] = emissions[0] / c * 0.5 * a1
      C_2[0] = emissions[0] / c * 0.5 * a2
      C_3[0] = emissions[0] / c * 0.5 * a3
      cumuptake[0] = emissions[0]
    
    
    #Set up maximal range of alpha sampling from which to linearly interpolate the answer
    alp = np.arange(0.0001,100.0,step=0.005)
    
    for x in range(1,t.size):
        
      #Calculate the parametrised iIRF and check if it is over the maximum allowed value
      iirf = (rc * cumuptake[x-1]  + rt* out_temp[x-1]  + r0   )
      if iirf >= iirf_max:
        iirf = iirf_max
      
      #Linearly interpolate a solution for alpha
      funct = alp*(a0*b0*(1.0 - np.exp(-t_iirf/(b0*alp))) + a1*b1*(1.0 - np.exp(-t_iirf/(b1*alp))) + a2*b2*(1.0 - np.exp(-t_iirf/(b2*alp))) + a3*b3*(1.0 - np.exp(-t_iirf/(b3*alp))))  -  iirf
      if np.sum(funct > 0.0) > 0:
        f = interp1d(x=funct,y=alp)
        time_scale_sf = f(0.0)
      else:
        print 'Max scaling factor reached: ',x
        time_scale_sf=alp[-1]

      #Multiply default timescales by scale factor
      b0_new = b0 * time_scale_sf
      b1_new = b1 * time_scale_sf
      b2_new = b2 * time_scale_sf
      b3_new = b3 * time_scale_sf
      

      #Do numerical method, carbon first followed by forcing followed by temperature
      C_0_d = C_0[x-1]*(np.exp((-1.0)/b0_new))
      C_1_d = C_1[x-1]*(np.exp((-1.0)/b1_new))
      C_2_d = C_2[x-1]*(np.exp((-1.0)/b2_new))
      C_3_d = C_3[x-1]*(np.exp((-1.0)/b3_new))

      C_0_f = 0.5*(a0)*(emissions[x-1]+emissions[x])/ c
      C_1_f = 0.5*(a1)*(emissions[x-1]+emissions[x])/ c
      C_2_f = 0.5*(a2)*(emissions[x-1]+emissions[x])/ c
      C_3_f = 0.5*(a3)*(emissions[x-1]+emissions[x])/ c

      C_0[x] = C_0_d + C_0_f
      C_1[x] = C_1_d + C_1_f
      C_2[x] = C_2_d + C_2_f
      C_3[x] = C_3_d + C_3_f
      out_concs[x]=C_0[x] + C_1[x] + C_2[x] + C_3[x]
      
      cumuptake[x] =  cumuptake[x-1] + emissions[x] - (out_concs[x] - out_concs[x-1])*c
      if type(other_rf) == float:
        radiative_forcing[x] = a * np.log((out_concs[x] + pre_indust_co2) /pre_indust_co2) + other_rf
      else:
        radiative_forcing[x] = a * np.log((out_concs[x] + pre_indust_co2) /pre_indust_co2) + other_rf[x]
    

      T_1_d = T_1[x-1]*(np.exp((-1.0)/d1))
      T_2_d = T_2[x-1]*(np.exp((-1.0)/d2))
      
      T_1_f = 0.5*(c1)*(radiative_forcing[x-1]+radiative_forcing[x])*(1-np.exp((-1.0)/d1))
      T_2_f = 0.5*(c2)*(radiative_forcing[x-1]+radiative_forcing[x])*(1-np.exp((-1.0)/d2))
      
      T_1[x] = T_1_d + T_1_f
      T_2[x] = T_2_d + T_2_f
      out_temp[x]=T_1[x] + T_2[x]
    
    if restart_out:
        restart_out_val=(C_0[-1],C_1[-1],C_2[-1],C_3[-1],T_1[-1],T_2[-1],cumuptake[-1])
        return out_concs + pre_indust_co2, out_temp, restart_out_val
    else:
        return out_concs + pre_indust_co2, out_temp




def fair_scm(emissions,other_rf=0.0,TCR=1.75,ECS=3.0,d1=4.1,d2=239.0,a0=0.2173,a1=0.2240,a2=0.2824,a3=0.2763,b0=1000000,b1=394.4,b2=36.54,b3=4.304,t_iirf=100.0,rt=4.5,temp_off=-0.5,r0=35.0,rc = 0.02,a=5.35,pre_indust_co2=278.0,c=2.123,iirf_max=97.0,restart_in=False,restart_out=False):

  t = np.arange(0.0,emissions.shape[-1])
  out_temp=np.zeros_like(emissions)
  out_concs=np.zeros_like(emissions)
  

  #Calculate the parameters of the temperature response model
  #c1 , c2 = inversion(TCR,ECS,d1=d1,d2=d2,a=a)
  c_therm = inversion(TCR,ECS,d1=d1,d2=d2,a=a)
  
  #Convert the emissions array into concentrations using the carbon cycle
  if restart_out:
    concs, temps, restart_out_val = integral_2_carbon_tempint(out_concs,out_temp,t,emissions,other_rf=other_rf,c1=c1,c2=c2,d1=d1,d2=d2,a0=a0,a1=a1,a2=a2,a3=a3,b0=b0,b1=b1,b2=b2,b3=b3,t_iirf=t_iirf,rt=rt,temp_off=temp_off,r0=r0,rc=rc,a=a,pre_indust_co2=pre_indust_co2,c=c,iirf_max=iirf_max,restart_in=restart_in,restart_out=restart_out)
    return out_temp, concs, restart_out_val
  else:
    concs, temps = integral_2_carbon_tempint(out_concs,out_temp,t,emissions,other_rf=other_rf,c1=c1,c2=c2,d1=d1,d2=d2,a0=a0,a1=a1,a2=a2,a3=a3,b0=b0,b1=b1,b2=b2,b3=b3,t_iirf=t_iirf,rt=rt,temp_off=temp_off,r0=r0,rc=rc,a=a,pre_indust_co2=pre_indust_co2,c=c,iirf_max=iirf_max,restart_in=restart_in,restart_out=restart_out)

    return out_temp, concs

def integral_2_carbon_emmsbac(out_concs_in,out_temp,t,emissions,other_rf,c1,c2,d1=4.1,d2=239.0,a0=0.2173,a1=0.2240,a2=0.2824,a3=0.2763,b0=1000000,b1=394.4,b2=36.54,b3=4.304,t_iirf=100.0,rt=4.5,temp_off=-0.5,r0=35.0,rc = 0.02,a=5.35,pre_indust_co2=278.0,c=2.123,iirf_max=97.0):
    #In this funciton the a values are the present-dat (2014) values that give current airborne fraction after 100 years
    out_concs = np.copy(out_concs_in) - pre_indust_co2

    #Do the integral in terms of the individual response boxes
    C_0 = np.zeros_like(out_concs)
    C_1 = np.zeros_like(out_concs)
    C_2 = np.zeros_like(out_concs)
    C_3 = np.zeros_like(out_concs)
    
    radiative_forcing = np.zeros_like(out_concs)
    cumuptake = np.zeros_like(out_concs)
    
    #INCLUDE INITIALISATION FOR MYLES METHOD
    cumuptake[0] = emissions[0]
    
    #Do the integral in terms of the individual response boxes
    out_temp[:] = 0.0
    T_1 = np.zeros_like(out_temp)
    T_2 = np.zeros_like(out_temp)
    
    emissions[:] = 0.0
    
    alp = np.arange(0.0001,100.0,step=0.005)
    
    for x in range(1,t.size):
      
      iirf = (rc * cumuptake[x-1]  + rt *out_temp[x-1]  + r0   )
      if iirf >= iirf_max:
        iirf = iirf_max
      funct = alp*(a0*b0*(1.0 - np.exp(-t_iirf/(b0*alp))) + a1*b1*(1.0 - np.exp(-t_iirf/(b1*alp))) + a2*b2*(1.0 - np.exp(-t_iirf/(b2*alp))) + a3*b3*(1.0 - np.exp(-t_iirf/(b3*alp))))  -  iirf

      if np.sum(funct > 0.0) > 0:
        f = interp1d(x=funct,y=alp)
        time_scale_sf = f(0.0)
      else:
          #print 'Max scaling factor reached'
        time_scale_sf=alp[-1]


      b0_new = b0 * time_scale_sf
      b1_new = b1 * time_scale_sf
      b2_new = b2 * time_scale_sf
      b3_new = b3 * time_scale_sf
      

      C_0_d = C_0[x-1]*(np.exp((-1.0)/b0_new))
      C_1_d = C_1[x-1]*(np.exp((-1.0)/b1_new))
      C_2_d = C_2[x-1]*(np.exp((-1.0)/b2_new))
      C_3_d = C_3[x-1]*(np.exp((-1.0)/b3_new))
 
      emissions[x]  = (c*(out_concs[x] - C_0_d - C_1_d - C_2_d - C_3_d) - 0.5*(a0+a1+a2+a3)*emissions[x-1]) / (0.5*(a0+a1+a2+a3))
      C_0_f = 0.5*(a0)*(emissions[x-1]+emissions[x])/ c
      C_1_f = 0.5*(a1)*(emissions[x-1]+emissions[x])/ c
      C_2_f = 0.5*(a2)*(emissions[x-1]+emissions[x])/ c
      C_3_f = 0.5*(a3)*(emissions[x-1]+emissions[x])/ c
      
      
      C_0[x] = C_0_d + C_0_f
      C_1[x] = C_1_d + C_1_f
      C_2[x] = C_2_d + C_2_f
      C_3[x] = C_3_d + C_3_f

      cumuptake[x] =  cumuptake[x-1] + emissions[x] - (out_concs[x] - out_concs[x-1])*c
      
      
      if type(other_rf) == float:
        radiative_forcing[x] = a * np.log((out_concs[x] + pre_indust_co2) /pre_indust_co2) + other_rf
      else:
        radiative_forcing[x] = a * np.log((out_concs[x] + pre_indust_co2) /pre_indust_co2) + other_rf[x]
    

      T_1_d = T_1[x-1]*(np.exp((-1.0)/d1))
      T_2_d = T_2[x-1]*(np.exp((-1.0)/d2))
      
      T_1_f = 0.5*(c1)*(radiative_forcing[x-1]+radiative_forcing[x])*(1-np.exp((-1.0)/d1))
      T_2_f = 0.5*(c2)*(radiative_forcing[x-1]+radiative_forcing[x])*(1-np.exp((-1.0)/d2))
      
      T_1[x] = T_1_d + T_1_f
      T_2[x] = T_2_d + T_2_f
      out_temp[x]=T_1[x] + T_2[x]

    return emissions, out_temp

def fair_scm_tempsback(temps_in,other_rf=0.0,TCR=1.75,ECS=3.0,d1=4.1,d2=239.0,a=5.35):

  out_temp = np.copy(temps_in)
  t = np.zeros_like(out_temp)
  T_1 = np.zeros_like(out_temp)
  T_2 = np.zeros_like(out_temp)
  
  co2_rf = np.zeros_like(out_temp)
  c1 , c2 = inversion(TCR,ECS,d1=d1,d2=d2,a=a)
  

  for x in range(1,t.size):
      
      T_1_d = T_1[x-1]*(np.exp((-1.0)/d1))
      T_2_d = T_2[x-1]*(np.exp((-1.0)/d2))
      
      co2_rf[x] = (out_temp[x] - T_1_d - T_2_d - 0.5*(co2_rf[x-1] + other_rf[x-1] )*(c1*(1-np.exp((-1.0)/d1)) + c2*(1-np.exp((-1.0)/d2))) - 0.5*other_rf[x]*(c1*(1-np.exp((-1.0)/d1)) + c2*(1-np.exp((-1.0)/d2)))) / (0.5*((c1*(1-np.exp((-1.0)/d1)) + c2*(1-np.exp((-1.0)/d2)))) )
      T_1_f = 0.5*(c1)*((co2_rf[x-1] + other_rf[x-1])+(co2_rf[x]+other_rf[x]))*(1-np.exp((-1.0)/d1))
      T_2_f = 0.5*(c2)*((co2_rf[x-1] + other_rf[x-1])+(co2_rf[x]+other_rf[x]))*(1-np.exp((-1.0)/d2))
      
      T_1[x] = T_1_d + T_1_f
      T_2[x] = T_2_d + T_2_f

  return co2_rf


def fair_scm_emmsback(co2_rf,other_rf=0.0,TCR=1.75,ECS=3.0,d1=4.1,d2=239.0,a0=0.2173,a1=0.2240,a2=0.2824,a3=0.2763,b0=1000000,b1=394.4,b2=36.54,b3=4.304,t_iirf=100.0,rt=4.5,r0=35.0,rc = 0.02,a=5.35,pre_indust_co2=275,c=2.123,iirf_max=97.0):


  t = np.arange(0.0,co2_rf.size)
  out_temp=np.zeros(co2_rf.size)

  emissions = np.zeros_like(co2_rf)
  
  out_concs = pre_indust_co2 * np.exp(co2_rf / a )
  

  #Calculate the parameters of the temperature response model
  c1 , c2 = inversion(TCR,ECS,d1=d1,d2=d2,a=a)
  
  #Convert the emissions array into concentrations using the carbon cycle
  emms, temps = integral_2_carbon_emmsbac(out_concs,out_temp,t,emissions,other_rf=other_rf,c1=c1,c2=c2,d1=d1,d2=d2,a0=a0,a1=a1,a2=a2,a3=a3,b0=b0,b1=b1,b2=b2,b3=b3,t_iirf=t_iirf,rt=rt,r0=r0,rc=rc,a=a,pre_indust_co2=pre_indust_co2,c=c,iirf_max=iirf_max)
  
  #emms = np.convolve(emms,np.ones((10,))/10.0,mode='same')

  return out_temp, emms







