####################################################
# FAIR - JULIA version (development)
# by Richard Millar
####################################################

import Images

function fair_scm(in_driver;other_rf=0.0,input_params=[1.75,2.5,4.1,239.0,0.2173,0.2240,0.2824,0.2763,1000000,394.4,36.54,4.304,100.0,35.0,0.02,4.5,3.74,278.0,2.123,97.0],restart_in=false,restart_out=false,mode="emissions_driven")

    #Broadcast everything to the same shape
    #Check that all parameter shapes are the same or 1
    if ndims(in_driver)==1
      e_dims = 1
      in_driver = reshape(in_driver,(1,size(in_driver)[1]))
    else
      e_dims = size(in_driver)[1]
    end
    t_dims = size(in_driver)[end]
    if ndims(input_params)==1
      p_dims = 1
      input_params = reshape(input_params,(1,size(input_params)[1]))
    else
      p_dims = size(input_params)[1]
    end


    #Add extra index to input timeseries array to allow it to span parameters dimension
    in_params = reshape(input_params,(1,size(input_params)...))

    #Name the specific parameters for readability
    ecstcr = in_params[:,:,1:2]
    d = in_params[:,:,3:4]
    a = in_params[:,:,5:8]
    b = in_params[:,:,9:12]
    t_iirf = in_params[:,:,13]
    r0 = in_params[:,:,14]
    rc = in_params[:,:,15]
    rt = in_params[:,:,16]
    F_2x = in_params[:,:,17]
    pre_indust_co2 = in_params[:,:,18]
    c = in_params[:,:,19]
    iirf_max = in_params[:,:,20]

    #calculate model coefficients from TCR and ECS parameters
    k = 1.0 - (d./70.0).*(1.0 - exp(-70.0./d))
    q = reshape(transpose((1.0 ./ F_2x) .* (1.0./(k[:,:,1]-k[:,:,2])) .* [ecstcr[:,:,1]-ecstcr[:,:,2].*k[:,:,2];ecstcr[:,:,2].*k[:,:,1]-ecstcr[:,:,1]]),(1,p_dims,2))

    #Form the output timeseries variables
    radiative_forcing = zeros((e_dims,p_dims,t_dims))
    cumuptake = zeros((e_dims,p_dims,t_dims))
    iirf = zeros((e_dims,p_dims,t_dims))
    R = zeros((e_dims,p_dims,t_dims,size(b)[end]))
    T = zeros((e_dims,p_dims,t_dims,size(q)[end]))

    
    #Reshape the input and output arrays dependent on the mode of operation
    if eltype(other_rf) != Float64
      if ndims(other_rf) == 1
        other_rf = reshape(other_rf,(1,size(other_rf)...))
      end
      other_rf = permutedims(reshape(other_rf,(1,size(other_rf)...)),[2,1,3])
    end
    if mode == "emissions_driven"
      emissions = permutedims(reshape(in_driver,(1,size(in_driver)...)),[2,1,3])
      out_concs = zeros((e_dims,p_dims,t_dims))
      out_temp = zeros((e_dims,p_dims,t_dims))
    end
    if mode == "emissions_back"
      out_concs = permutedims(reshape(in_driver,(1,size(in_driver)...)),[2,1,3]) .- reshape(pre_indust_co2,(size(pre_indust_co2)...,1))
      emissions = zeros((e_dims,p_dims,t_dims))
      out_temp = zeros((e_dims,p_dims,t_dims))
    end
    if mode == "forcing_driven"
      radiative_forcing = permutedims(reshape(in_driver,(1,size(in_driver)...)),[2,1,3])
      out_temp = zeros((e_dims,p_dims,t_dims))
      out_concs = zeros((e_dims,p_dims,t_dims))
    end
    if mode == "forcing_back"
      out_temp = permutedims(reshape(in_driver,(1,size(in_driver)...)),[2,1,3])
      radiative_forcing = zeros((e_dims,p_dims,t_dims))
      out_concs = zeros((e_dims,p_dims,t_dims))
    end

    #Load in the restart state if one is given
    if restart_in==true
      R[:,:,1,:]=restart_in[1]
      T[:,:,1,:]=restart_in[2]
      cumuptake[:,:,1] = restart_in[3]
      if mode != "forcing_back"
        out_temp[:,:,1] = sum(T[:,:,1,:],ndims(T[:,:,1,:]))
      end
      if mode != "emissions_back"
        out_concs[:,:,1] = sum(R[:,:,1,:],ndims(R[:,:,1,:]))
      end
      if mode != "emissions_driven"
        emissions[:,:,1] = restart_in[4]
      end
      if mode != "forcing_driven"
        if eltype(other_rf) == Float64
          radiative_forcing[:,:,1] = (F_2x./log(2.)) .* log((out_concs[:,:,1] + pre_indust_co2) ./pre_indust_co2) + other_rf
        else
          radiative_forcing[:,:,1] = (F_2x./log(2.)) .* log((out_concs[:,:,1] + pre_indust_co2) ./pre_indust_co2) + other_rf[:,:,1]
        end
      end
    
    elseif (mode in ["emissions_driven","emissions_back"])
        #Initialise the carbon pools to be correct for first timestep in numerical method
        R[:,:,1,:] = a .* reshape(emissions[:,:,1],(size(emissions[:,:,1])...,1)) ./ (reshape(c,(size(c)...,1))) * 0.5
        cumuptake[:,:,1] = repeat(emissions[:,:,1],outer=[1,p_dims,1])
    end

    #Do the main timestepping loop over the final dimension of the arrays
    for x in [2:(size(in_driver)[end])]
      if (mode in ["emissions_driven","emissions_back"])

        #Calculate IIRF
        iirf[:,:,x] = (rc .* cumuptake[:,:,x-1]  + rt.* out_temp[:,:,x-1]  .+ r0   )
        logic = falses(iirf)
        logic[:,:,x] = (iirf[:,:,x].-iirf_max).>0
        iirf[logic] = repeat(iirf_max,outer=[e_dims,1])[(iirf[:,:,x].-iirf_max).>0]

        #####################################################################
        #Linearly interpolate a solution for alpha NB/ NOT SURE HOW TO DO THIS IN JULIA
        #if x == 1:
        #  time_scale_sf = (root(iirf_interp_funct,0.16*np.ones((e_dims,p_dims)),args=(a,b,t_iirf,iirf,x,e_dims,p_dims)))['x'].reshape(e_dims,p_dims)
        #else:
        #  time_scale_sf = (root(iirf_interp_funct,time_scale_sf,args=(a,b,t_iirf,iirf,x,e_dims,p_dims)))['x'].reshape(e_dims,p_dims)
        #######################################################################

        #Multiply default timescales by scale factor
        time_scale_sf = ones((e_dims,p_dims))  #Set as ones until the code above is translated
        b_new = b .* reshape(time_scale_sf,(size(time_scale_sf)...,1))

      end
      
      if mode == "emissions_driven"
        #Do forward numerical method, carbon first
        R[:,:,x,:] = R[:,:,x-1,:].*permutedims(reshape((exp((-1.0)./b_new)),(1,size((exp((-1.0)./b_new)))...)),[2,3,1,4])
        add = 0.5*(a).*(reshape(emissions[:,:,x-1],(size(emissions[:,:,x-1])...,1))+reshape(emissions[:,:,x],(size(emissions[:,:,x])...,1)))./ reshape(c,(size(c)...,1))
        R[:,:,x,:] = R[:,:,x,:] + permutedims(reshape(add,(1,size(add)...)),[2,3,1,4])

        out_concs[:,:,x] = sum(R[:,:,x,:],ndims(R[:,:,x,:]))
        cumuptake[:,:,x] =  cumuptake[:,:,x-1] + repeat(emissions[:,:,x],outer=[1,p_dims,1]) - (out_concs[:,:,x] - out_concs[:,:,x-1]).*c
      end

      if mode == "emissions_back":
        #Back calculate required emissions and then update carbon pool arrays
        R[:,:,x,:] = R[:,:,x-1,:].*permutedims(reshape((exp((-1.0)./b_new)),(1,size((exp((-1.0)./b_new)))...)),[2,3,1,4])
        emissions[:,:,x]  = (c.*(out_concs[:,:,x] - sum(R[:,:,x,:],ndims(R[:,:,x,:]))[:,:,:,1]) - 0.5*sum(a,ndims(a)).*repeat(emissions[:,:,x-1],outer=[1,p_dims])) ./ (0.5*sum(a,ndims(a)))
        add = 0.5*(a).*(reshape(emissions[:,:,x-1],(size(emissions[:,:,x-1])...,1))+reshape(emissions[:,:,x],(size(emissions[:,:,x])...,1)))./ reshape(c,(size(c)...,1))
        R[:,:,x,:] = R[:,:,x,:] + permutedims(reshape(add,(1,size(add)...)),[2,3,1,4])
        cumuptake[:,:,x] =  cumuptake[:,:,x-1] + emissions[:,:,x] - (out_concs[:,:,x] - out_concs[:,:,x-1]).*c
      end

      #Calculate the radiative forcing for the thermal model
      if (mode in ["emissions_driven","emissions_back"])
        if eltype(other_rf) == Float64
            
          #NB/ CALC OTHER RFS COULD BE DONE IN HERE INTERACTIVLY BY CALLING BOX MODEL FUNCTIONS
          radiative_forcing[:,:,x] = (F_2x/log(2.)) .* log((out_concs[:,:,x] .+ pre_indust_co2) ./pre_indust_co2) + other_rf
        else
          radiative_forcing[:,:,x] = (F_2x/log(2.)) .* log((out_concs[:,:,x] .+ pre_indust_co2) ./pre_indust_co2) + repeat(other_rf[:,:,x],outer=[1,p_dims])
        end
      end

      if (mode in ["emissions_driven","emissions_back","forcing_driven"])
        #Thermal model numerical method
        T[:,:,x,:] = T[:,:,x-1,:].*permutedims(reshape((exp((-1.0)./d)),(1,size((exp((-1.0)./d)))...)),[2,3,1,4])
        add = 0.5*(q).*(reshape(radiative_forcing[:,:,x-1],(size(radiative_forcing[:,:,x-1])...,1))+reshape(radiative_forcing[:,:,x],(size(radiative_forcing[:,:,x])...,1))).*(1-exp((-1.0)./d))
        T[:,:,x,:] = T[:,:,x,:] + permutedims(reshape(add,(1,size(add)...)),[2,3,1,4])
        out_temp[:,:,x]=sum(T[:,:,x,:],ndims(T[:,:,x,:]))
      end

      if mode == "forcing_back"
        #Back calculate the forcing needed to get a temperature timeseries
        T[:,:,x,:] = T[:,:,x-1,:].*permutedims(reshape((exp((-1.0)./d)),(1,size((exp((-1.0)./d)))...)),[2,3,1,4])
        radiative_forcing[:,:,x]  = (out_temp[:,:,x] - sum(T[:,:,x,:],ndims(T[:,:,x,:]))[:,:,:,1] - 0.5*sum(q.*(1-exp((-1.0)./d)),ndims(q.*(1-exp((-1.0)./d)))).*radiative_forcing[:,:,x-1]) ./ (0.5*sum(q.*(1-exp((-1.0)./d)),ndims(q.*(1-exp((-1.0)./d)))))
        add = 0.5*(q).*(reshape(radiative_forcing[:,:,x-1],(size(radiative_forcing[:,:,x-1])...,1))+reshape(radiative_forcing[:,:,x],(size(radiative_forcing[:,:,x])...,1))).*(1-exp((-1.0)./d))
        T[:,:,x,:] = T[:,:,x,:] + permutedims(reshape(add,(1,size(add)...)),[2,3,1,4])
        out_temp[:,:,x]=sum(T[:,:,x,:],ndims(T[:,:,x,:]))
      end

      end


    #Prepare the output and return
    return_concs = out_concs .+ reshape(pre_indust_co2,size(pre_indust_co2)...,1)
    return_temps = out_temp
    return_emissions = emissions
    return_forcing = radiative_forcing
    if e_dims==1 & p_dims!=1
      return_concs = return_concs[1]
      return_temps = return_temps[1]
      return_emissions = emissions[1]
      return_forcing = radiative_forcing[1]
    end
    if e_dims!=1 & p_dims==1
      return_concs = return_concs[:,1]
      return_temps = return_temps[:,1]
      return_emissions = emissions[:,1]
      return_forcing = radiative_forcing[:,1]
    end
    if e_dims==1 & p_dims==1
      return_concs = return_concs[1,1]
      return_temps = return_temps[1,1]
      return_emissions = emissions[1,1]
      return_forcing = radiative_forcing[1,1]
      
    #Get the required output
    if mode == "emissions_driven"
      if restart_out==true
        restart_out_val=(R[:,:,end,:],T[:,:,end,:],cumuptake[:,:,end])
        return [return_concs, return_temps], restart_out_val
      else
        return [return_concs, return_temps]
      end
    end
    if mode == "emissions_back"
      #Smooth the backed out emissions NB/ NOT SURE HOW TO DO THIS IN JULIA
      #emissions = Images.imfilter_gaussian(emissions, [0.0, 0.0, 2.0])
      if restart_out==true
        restart_out_val=(R[:,:,end,:],T[:,:,end,:],cumuptake[:,:,end])
        return [return_emissions, return_temps], restart_out_val
      else
        return [return_emissions, return_temps]
      end
    end
    if mode == "forcing_driven"
      if restart_out==true
        restart_out_val=(R[:,:,end,:],T[:,:,end,:],cumuptake[:,:,end])
        return return_temps, restart_out_val
      else
        return return_temps
      end
    if mode == "forcing_back"
      #Smooth the backed out emissions NB/ NOT SURE HOW TO DO THIS IN JULIA
      #radiative_forcing = gaussian_filter1d(radiative_forcing,sigma=2,axis=-1)
      if restart_out==true
        restart_out_val=(R[:,:,end,:],T[:,:,end,:],cumuptake[:,:,end],emissions[:,:,end])
        return return_forcing, restart_out_val
      else:
        return return_forcing
      end
    end





