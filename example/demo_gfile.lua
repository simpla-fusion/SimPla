Description="For Cold Plasma Dispersion" -- description or other text things.    

Context="ExplicitEM"

-- SI Unit System
c = 299792458  -- m/s
qe=1.60217656e-19 -- C
me=9.10938291e-31 --kg
mp=1.672621777e-27 --kg
mp_me=1836.15267245 --
KeV = 1.1604e7    -- K
Tesla = 1.0       -- Tesla    
PI=3.141592653589793
TWOPI=PI*2
k_B=1.3806488e-23 --Boltzmann_constant
--

k_parallel=18
Btor= 1.0  * Tesla
Ti =  0.03 * KeV
Te =  0.05 * KeV
N0 = 1.0e17 -- m^-3


omega_ci = qe * Btor/mp -- e/m_p B0 rad/s
vTi= math.sqrt(k_B*Ti*2/mp)
rhoi = vTi/omega_ci    -- m

omega_ce = qe * Btor/me -- e/m_p B0 rad/s
vTe= math.sqrt(k_B*Te*2/me)
rhoe = vTe/omega_ce    -- m

NX = 200
NY = 200
NZ = 1
LX = 1.6 --m --100000*rhoi --0.6
LY = 2.8 --2.0*math.pi/k0
LZ = 0 -- 2.0*math.pi/18
GW = 5 

omega_ext=omega_ci*1.9


-- From Gan
--[[
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
--]]


InitN0=function(x,y,z)      
      local x0=0.1*LX ;
      local res = 0.0;
      if x>x0 then
        res=0.5*N0*(1.0- math.cos(PI*(x-x0)/(LX-x0)));
      end
      return res
     end 


InitB0=function(x,y,z)
      local X0 = 12*LX/NX;
      local DEN_JUMP = 0.4*LX;
      local DEN_GRAD = 0.2*LX;
      local AtX0 = 2./math.pi*math.atan((-DEN_JUMP)/DEN_GRAD);
      local AtLX = 2./math.pi*math.atan((LX-DEN_JUMP-X0)/DEN_GRAD);
      local DenCof = 1./(AtLX-AtX0);
      local dens1 = DenCof*(2./math.pi*math.atan((x-DEN_JUMP)/DEN_GRAD)-AtX0);
      return {0,0,Btor}
     end 



--[[
InitValue={

  E=function(x,y,z)
     ---[[
      local res = 0.0;
      for i=1,40 do
          res=res+math.sin(x/LX*TWOPI* i);
      end;
    
      return {res,res,res}
    end
    ,

  B=function(x,y,z)
      -- local omega_ci_x0 = 1/1.55*omega;
      -- local omega_ci_lx = 1/1.45*omega;
      -- local Bf_lx = omega_ci_lx*ionmass/ioncharge
      -- local Bf_x0 = omega_ci_x0*ionmass/ioncharge

      return {0,0,Btor}  
     end
     ,

}
--]]
     
GFile='/home/salmon/workspace/SimPla/example/g033068.02750'
Grid=
{
  Type="RectMesh",

  UnitSystem={Type="SI"},
  Topology=
  {       
      Type="RectMesh",
      Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid
      ArrayOrder="Fortran Order"       
      
  },
  Geometry=
  {
      Type="Origin_DxDyDz",
      Min={1.2,-1.4,0.0},
      Max={2.8,1.4,LZ},
      -- dt= 2.0*math.pi/omega_ci/1000.0
      dt=0.5*LX/NX/c  -- time step     
  },
  
}
 
-- Media=
-- {
--    {Tag="Vacuum",Region={{0.2*LX,0,0},{0.8*LX,0,0}},Op="Set"},

--    {Tag="Plasma",
--      Select=function(x,y,z)
--           return x>1.0 and x<2.0
--         end
--      ,Op="Set"},
-- }
 
---[[
Constraints=
{
  { 
    DOF="E",IsHard=true,
	Select={Type="Boundary", Material="Vacuum" },
	Value= 0
  },
  { 
    DOF="J",
	Index={ {NX/2,NY/2,0}},
	IsHard=false,
  	Value=function(x,y,z,t)	
      local tau = t*omega_ext*100
      return {0, math.sin(tau) ,0}   
      end
	 
  } 
-- *(1-math.exp(-tau*tau)
   
 --  { 
 --    DOF="J",
	-- Select={Type="Media", Tag="Vacuum"},
	-- Value= 0
 --  },
 --  { 
 --    DOF="Particles",
	-- Select={Type="Media", Tag="Vacuum"},
	-- Value= "Absorb"
 --  },
   
}
--]]


--[[
Particles={
  {Name="H",Engine="DeltaF",Mass=mp,Charge=qe,T=Ti,PIC=100, n=InitN0}
}

FieldSolver= 
{

   ColdFluid=
    {
       Nonlinear=false,       
       Species=
       {
      -- {Name="ion",Mass=mp,  Charge= qe ,T=Ti,  n=InitN0, J=0},
       {Name="ele",Mass=me,  Charge=-qe,   n=InitN0, J=0}         
        }
    },

--  PML=  {Width={20,20,0,0,0,0}}
}
--]]
 

-- The End ---------------------------------------

