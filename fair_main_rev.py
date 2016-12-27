import numpy as np

def integral_2_carbon_tempint(out_concs,out_temp,t,emissions,other_rf,c1,c2,d1=4.1,d2=239.0,a0=0.2173,a1=0.2240,a2=0.2824,a3=0.2763,b0=1000000,b1=394.4,b2=36.54,b3=4.304,t_iirf=100.0,rt=4.5,temp_off=-0.5,r0=35.0,rc = 0.02,a=5.35,pre_indust_co2=278.0,c=2.123,iirf_max=97.0,restart_in=False,restart_out=False):

    #specific TCR and ECS
    k = 1.0 - (d/70.0)*(1.0 - np.exp(-70.0/d))
    q = np.transpose((1.0 / F_2x) * (1.0/(k[:,0]-k[:,1])) * np.array([ecstcr[:,0]*k[:,0]-ecstcr[:,1],ecstcr[:,1]-ecstcr[:,0]*k[:,1]]) )

    #Do the integral in terms of the individual response boxes
    out_concs = np.zeros_like(emissions)
    R = np.zeros(shape=(emissions.shape[0],b.shape[-1],emissions.shape[1]))
    radiative_forcing = np.zeros_like(emissions)
    cumuptake = np.zeros_like(emissions)
    iirf = np.zeros_like(emissions)
    
    #Initialise the temperature anomaly boxes
    out_temp = np.zeros_like(emissions)
    T = np.zeros(shape=(emissions.shape[0],d.shape[-1],emissions.shape[1]))

    
    if restart_in:
      R[:,:,0]=restart_in[0]
      T[:,:,0]=restart_in[1]
      cumuptake[:,0] = restart_in[2]
    
    else:              

      #Initialise the carbon pools to be correct for first timestep in numerical method
      R[:,:,0] = emissions[:,0,np.newaxis] / p * 0.5 * a
      cumuptake[:,0] = emissions[:,0]
    
    
    #Set up maximal range of alpha sampling from which to linearly interpolate the answer
    alp = np.arange(0.0001,100.0,step=0.005)
    
    for x in range(1,emissions.shape[-1]):
        
      #Calculate the parametrised iIRF and check if it is over the maximum allowed value
      iirf[:,x] = (rc * cumuptake[:,x-1]  + rt* out_temp[:,x-1]  + r0   )
      iirf[np.where(iirf[:,x]>=iirf_max),x] = iirf_max
      
      #Linearly interpolate a solution for alpha
      funct = alp[:,np.newaxis]*np.sum(a*b*(1.0 - np.exp(-t_iirf/(b*alp[:,np.newaxis]))),axis=0)   -  iirf
      if np.sum(funct > 0.0) > 0:
        f = interp1d(x=funct,y=alp)
        time_scale_sf = f(0.0)
      else:
        print 'Max scaling factor reached: ',x
        time_scale_sf=alp[-1]

      #Multiply default timescales by scale factor
      b_new = b * time_scale_sf
      

      #Do numerical method, carbon first followed by forcing followed by temperature
      R[:,:,x] = R[:,:,x-1]*(np.exp((-1.0)/b_new)) + 0.5*(a)*(emissions[:,x-1,np.newaxis]+emissions[:,x,np.newaxis])/ p
      out_concs[:,x] = np.sum(R[:,:,x],axis=0)
      cumuptake[:,x] =  cumuptake[:,x-1] + emissions[:,x] - (out_concs[:,x] - out_concs[:,x-1])*p
      
      if type(other_rf) == float:
        radiative_forcing[:,x] = a * np.log((out_concs[:,x] + pre_indust_co2) /pre_indust_co2) + other_rf[:,x]
      else:
        radiative_forcing[:,x] = a * np.log((out_concs[:,x] + pre_indust_co2) /pre_indust_co2) + other_rf[:,x]
    

      T[:,x] = T[:,x-1]*(np.exp((-1.0)/d)) + 0.5*(q)*(radiative_forcing[:,x-1,np.newaxis]+radiative_forcing[:,x,np.newaxis])*(1-np.exp((-1.0)/d))

      
      out_temp[:,x]=np.sum(T[:,x],axis=0)
    
    if restart_out:
        restart_out_val=(R[:,:,-1],T_1[:,:,-1],cumuptake[:,-1])
        return out_concs + pre_indust_co2, out_temp, restart_out_val
    else:
        return out_concs + pre_indust_co2, out_temp

