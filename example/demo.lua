-- Auto-generated  config file.------------------- --------
-- Jun 25 2011 14:55:06

    Description="For Cold Plasma Dispersion" -- description or other text things.    


    UnitSystem = 
     { 
       type= "SI"
     }
     
    
    c = 3.1415926e8 -- m/s
    Btor= 1.0 -- Tesla
    Ti =  0.3 -- KeV
    Te =  0.1 -- KeV
    rhoi = 4.57*1e-3 * math.sqrt(Ti)/Btor  --1.02 * math.sqrt(Ti)/(1e4*Btor) -- m
--    print(rhoi)

    k0 = 25./40.
    NX = 411
    NY = 1
    NZ = 1
    LX = 2.0*math.pi/k0   --2.0*math.pi*1000 * rhoi --0.6
    LY = 2.0*math.pi
    LZ = 2.0*math.pi  -- 2.0*math.pi/18
    GW = 5 
    n0 = 1.07e17 -- 4*Btor*Btor* 5.327934360e15 -- m^-3
    omega_ci = 9.58e7*Btor -- rad/s

    
    
    Mesh={
             
      dims={NX,NY,NZ}, -- number of grid, now only first dimension is valid       
	      
      gw= {5,1,1},  -- width of ghost points            
      
      xmin={0,0,0},
      
      xmax={LX/(NX-GW*2-1)*(NX-1),LY,LZ}    ,       
	      
      dt=0.5*LX/ (NX-1)/c  -- time step     
    }
             
    DIAGNOSIS={"E1","B1"}
    
    SP_LIST= {"HG"} -- the list of species in the simulation        
             

    N0={}  -- initial/background density field of  electron;         
    B0={}
    E0={}

	  
    for x=0,DIMS[1] -1 do        
     for y=0,DIMS[2] -1 do     
      for z=0,DIMS[3] -1 do    
         s=x*DIMS[2]*DIMS[3]+ y*DIMS[3]+z   
         -- if x> (DIMS[1]-1)/10 and x <(DIMS[1]-1)/10*9  then 
         --  r=(x-(DIMS[1]-1)*0.1)/(DIMS[1]-1)*10/8.0
         --  N0[s]= n0 *(1-4*(r-0.5)*(r-0.5))
         -- end
         N0[s] = n0
         r=x/(DIMS[1]-1)
         B0[s*3+2]=Btor -- *1.2*(1/2.3+(1/1.8-1/2.3)*r)  
      end    
     end    
    end     
    
    LOAD_FIELD=
    
    SPECIES= -- predefine species         
    {                
         ele	={desc="ele" ,Z=-1.0,   m=1.0/100.0, engine="ColdFluid", Ts=0.0},
   
         eleG	={desc="ele" ,Z=-1.0,	m=1.0/100.0, engine="GyroGauge", Ts=Te, numOfMate=4,PIC=20 },
             
         HC		={desc="H"    ,Z=1.0,   m=1, 	engine="ColdFluid", 	Ts=0.0},       
             
         HG		={desc="H"    ,Z=1.0,   m=1, 	engine="GyroGauge", Ts=Ti, numOfMate=20,    PIC=20},
             
         HD		={desc="H"    ,Z=1.0,   m=1,  	engine="DeltaF", 	Ts=Ti, numOfMate=1,     PIC=200},
          
    } 

   ---[[ uncomment this line, if you need Cycle BC.
    -- set BC(boundary condition), now only first two are valid         
    -- BC >= GW               
    BC= {             
        0,0, -- direction x  
        0,0, -- direction y  
        0,0 -- direction z  
       	}            
       	
    for x=0,DIMS[1] -1 do         
     for y=0,DIMS[2] -1 do    
      for z=0,DIMS[3] -1 do     
        s=x*DIMS[2]*DIMS[3]+ y*DIMS[3]+z    
        qx = (x-GW[1])%(DIMS[1]-2*GW[1]-1)/(DIMS[1]-2*GW[1]-1) *2.0 *math.pi    
        qy = (y-GW[2])%(DIMS[2]-2*GW[2]-1)/(DIMS[2]-2*GW[2]-1) *2.0 *math.pi    
        qz = (z-GW[3])%(DIMS[3]-2*GW[3]-1)/(DIMS[3]-2*GW[3]-1) *2.0 *math.pi    
        a=0.0    
        for kx=1,40 do    
         for ky=1,1 do    
            a=a+math.sin(qx*kx)*math.sin(qy*ky)    
         end    
        end    
        E0[s*3+0]=a  
      end    
     end    
    end    
    --]]            
             
    --[[ uncomment this line, if you need PML BC. 
    BC= {            
         13,25, -- direction x 
         0,0, -- direction y 
         0,0 -- direction z 
         }            
    Srcf0=1.2*omega_ci    
    Srckx=0            
    Srctau=0.5*Srcf0         
             
    function JSrc(t)              
         alpha= Srctau*t      
         res={}               
         if alpha<0 then              
               a=1-math.exp(-alpha*alpha)       
         else               
           a=1          
         end         
    
        for z =  0,DIMS[3]-1 do
           s = 15 * DIMS[2]*DIMS[3] + (DIMS[2]-1)*DIMS[3] + z
           res[s*3+2] = math.sin(math.pi*2.0*(z/(DIMS[3]-1-GW[3]*2)) + Srcf0*t)*a*1e-8 
        end
               
        return res           
    end                
    --]]            
-- The End ---------------------------------------

