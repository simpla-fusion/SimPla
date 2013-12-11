Description="For Cold Plasma Dispersion" -- description or other text things.    
-- SI Unit System
c = 3.1415926e8   -- m/s
KeV = 1.1604e7    -- K
Tesla = 1.0       -- Tesla    
--

Btor= 1.2  * Tesla
Ti =  0.03 * KeV
Te =  0.05 * KeV

rhoi = 4.57*1e-3 * math.sqrt(Ti)/Btor  --1.02 * math.sqrt(Ti)/(1e4*Btor) -- m
--    print(rhoi)

k0 = 25./40.
NX = 600
NY = 1
NZ = 1
LX = 0.06   --2.0*math.pi*1000 * rhoi --0.6
LY = 0
LZ = 0  -- 2.0*math.pi/18
GW = 5 
N0 = 0.8*0.25e18 -- 4*Btor*Btor* 5.327934360e15 -- m^-3

omega_ci = 95790338.4 * Btor -- e/m_p B0 rad/s

-- From Gan

InitValue={
  n0=function(x,y,z)
      local X0 = 12*LX/NX;
      local DEN_JUMP = 0.4*LX;
      local DEN_GRAD = 0.2*LX;
      local AtX0 = 2./math.pi*math.atan((-DEN_JUMP)/DEN_GRAD);
      local AtLX = 2./math.pi*math.atan((LX-DEN_JUMP-X0)/DEN_GRAD);
      local DenCof = 1./(AtLX-AtX0);
      local dens1 = DenCof*(2./math.pi*math.atan((x-DEN_JUMP)/DEN_GRAD)-AtX0);
      return dens1*N0
     end   
     ,
  B0=function(x,y,z)
--[[  
      local omega_ci_x0 = 1/1.55*omega;
      local omega_ci_lx = 1/1.45*omega;
      local Bf_lx = omega_ci_lx*ionmass/ioncharge
      local Bf_x0 = omega_ci_x0*ionmass/ioncharge
--]]
      return Btor  
     end
      ,
  E0=0.0, E1=0.0, B1=0.0,J1=0.0
}

Grid=
{
  Type="CoRectMesh",
  UnitSystem={Type="SI"},
  Topology=
  {       
      Type="3DCoRectMesh",
      Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid       
      GhostWidth= {GW,GW,GW},  -- width of ghost points            
  },
  Geometry=
  {
      Type="Origin_DxDyDz",
      Min={0.0,0.0,0.0},
      Max={LX,LY,LZ},
      dt=0.5*LX/ (NX-1)/c  -- time step     
  }
}

FieldSolver= 
  {
       Type="Default"
       -- Type="PML",  bc={5,5,5,5,5,5}
  }
Particles=
 {
     {Name="ion",Mass=1.0,Charge=1.0,Engine="ColdFluid",T= Ti},
     {Name="ele",Mass=1.0/1836.2,Charge=-1.0,Engine="ColdFluid",T=Te}         
 }
CurrentSrc=
 { 
   x={0.0,0.0,0.0},
   J= function(t)
        local tau = t*omega_ci
        return math.sin(tau)*(1-math.exp(-0.01*tau*tau))   
      end


 }
   --[[ uncomment this line, if you need Cycle BC.
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

