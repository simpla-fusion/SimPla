Description="For Cold Plasma Dispersion" -- description or other text things.    
-- SI Unit System
c = 299792458  -- m/s
e=1.60217656e-19 -- C
me=9.10938291e-31 --kg
mp=1.672621777e-27 --kg
mp_me=1836.15267245 --
KeV = 1.1604e7    -- K
Tesla = 1.0       -- Tesla    
PI=3.141592653589793
TWOPI=PI*2
k_B=1.3806488e-23 --Boltzmann_constant
--

Btor= 1.0  * Tesla
Ti =  0.03 * KeV
Te =  0.05 * KeV
N0 = 1.0e18 -- m^-3


omega_ci = e * Btor/mp -- e/m_p B0 rad/s
vTi= math.sqrt(k_B*Ti*2/mp)
rhoi = vTi/omega_ci    -- m

omega_ce = e * Btor/me -- e/m_p B0 rad/s
vTe= math.sqrt(k_B*Te*2/me)
rhoe = vTe/omega_ce    -- m

NX = 200
NY = 1
NZ = 1
LX = 100*rhoi --0.6
LY = 0 --2.0*math.pi/k0
LZ = 0  -- 2.0*math.pi/18
GW = 5 



-- From Gan

InitN0=function(x,y,z)
      local X0 = 12*LX/NX;
      local DEN_JUMP = 0.4*LX;
      local DEN_GRAD = 0.2*LX;
      local AtX0 = 2./math.pi*math.atan((-DEN_JUMP)/DEN_GRAD);
      local AtLX = 2./math.pi*math.atan((LX-DEN_JUMP-X0)/DEN_GRAD);
      local DenCof = 1./(AtLX-AtX0);
      local dens1 = DenCof*(2./math.pi*math.atan((x-DEN_JUMP)/DEN_GRAD)-AtX0);
      return dens1*N0
     end 

InitValue={ 
  E=function(x,y,z)
     ---[[
      local res = 0.0;
      for i=1,20 do
          res=res+math.sin(x/LX*TWOPI* i);
      end;
    -- ]]
      return {0,0,res}
    end

  , J=0.0

}

B_background=function(x,y,z)
--[[  
      local omega_ci_x0 = 1/1.55*omega;
      local omega_ci_lx = 1/1.45*omega;
      local Bf_lx = omega_ci_lx*ionmass/ioncharge
      local Bf_x0 = omega_ci_x0*ionmass/ioncharge
--]]
      return {0,0,Btor}  
     end
      

-- GFile
Grid=
{
  Type="CoRectMesh",
--  ScalarType="Complex",
  UnitSystem={Type="SI"},
  Topology=
  {       
      Type="3DCoRectMesh",
      Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid       
      
  },
  Geometry=
  {
      Type="Origin_DxDyDz",
      Min={0.0,0.0,0.0},
      Max={LX,LY,LZ},
--      dt= 2.0*math.pi/omega_ci/100.0
      dt=0.5*LX/NX/c  -- time step     
  },
  
}
--[[
Media=
{
   {Type="Vacuum",Region={{0.2*LX,0,0},{0.8*LX,0,0}},Op="Set"},

   {Type="Plasma",
     Select=function(x,y,z)
          return x>1.0 and x<2.0
        end
     ,Op="Set"},
}

Boundary={
   --   { Type="PEC", In="Vacuum",Out="NONE"},
   -- { Type="PEC", In="Plasma",Out="NONE"},
}
--]]

Particles={
--  {Name="H",Engine="Full",m=1.0,Z=1.0,PIC=100}
}

FieldSolver= 
{
---[[
   ColdFluid=
    {
    B0 = {0,0,Btor},
       {Name="ion",m=1.0,       Z= 1.0,T=Ti,  n=N0, J=0},
       {Name="ele",m=1.0/1836.2,Z=-1.0,T=Te,  n=N0, J=0}         
    },
--]]
--    PML=  {Width={8,8,0,0,0,0}}
}

--[[
CurrentSrc=
 { 
  
  Points={{0.1*LX,0.0,0.0},},
  Fun=function(x,y,z,t)
      local tau = t*omega_ci*1.9
      return {0,math.sin(tau)*(1-math.exp(-tau*tau)),0}   
      end
 }
]]

-- The End ---------------------------------------

