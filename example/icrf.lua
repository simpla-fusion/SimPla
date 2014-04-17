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
epsilon0=8.8542e-12
--

k_parallel=6.5
Btor= 1.0  * Tesla
Ti =  0.0003 * KeV
Te =  0.05 * KeV
N0 = 1.0e19-- m^-3


omega_ci = e * Btor/mp -- e/m_p B0 rad/s
vTi= math.sqrt(k_B*Ti*2/mp)
rhoi = vTi/omega_ci    -- m

omega_ce = e * Btor/me -- e/m_p B0 rad/s
vTe= math.sqrt(k_B*Te*2/me)
rhoe = vTe/omega_ce    -- m
omeaga_pe=math.sqrt(N0*e*e/(me*epsilon0))

NX = 128
NY = 1
NZ = 1
LX = 0.05 --m --100000*rhoi --0.6
LY = 0 --2.0*math.pi/k0
LZ = 0 -- 2.0*math.pi/18
GW = 5 

omega_ext=omega_ci*1.2


-- From Gan
InitN0=function(x,y,z)
      -- local X0 = 12*LX/NX;
      -- local DEN_JUMP = 0.4*LX;
      -- local DEN_GRAD = 0.2*LX;
      -- local AtX0 = 2./math.pi*math.atan((-DEN_JUMP)/DEN_GRAD);
      -- local AtLX = 2./math.pi*math.atan((LX-DEN_JUMP-X0)/DEN_GRAD);
      -- local DenCof = 1./(AtLX-AtX0);
      -- local dens1 = DenCof*(2./math.pi*math.atan((x-DEN_JUMP)/DEN_GRAD)-AtX0);
      return N0 --*dens1
     end 

InitB0=function(x,y,z)
      return {0,0,Btor}
     end 
 


InitValue={

   E=function(x,y,z)
     ---[[
      local res = 0.0;
      for i=1,20 do
          res=res+math.sin(x/LX*TWOPI* i);
      end;
    
      return {res,res,res}
    end

 --[[    
--]]
 --   E 	= 0.0 
  , J 	= 0.0
  , B 	= InitB0
  , ne 	= InitN0

}

 

Grid=
{
  Type="RectMesh",
 
  UnitSystem={Type="SI"},

  Topology=
  {       
      Type="RectMesh",
      Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid       
      
  },
  Geometry=
  {
      Type="Origin_DxDyDz",
      Min={0.0,0.0,0.0},
      Max={LX,LY,LZ},
     --dt= 2.0*math.pi/omega_ci/100.0
   
  },
     -- dt= TWOPI/omeaga_pe
     dt= 0.5*LX/NX/c -- time step     
}
--[[
Model=
{
   {Type="Vacuum",Region={{0.2*LX,0,0},{0.8*LX,0,0}},Op="Set"},

   {Type="Plasma",
     Select=function(x,y,z)
          return x>1.0 and x<2.0
        end
     ,Op="Set"},
}
--]]

 
---[[ 
Particles={
-- H ={Type="Full",Mass=mp,Charge=e,Temperature=Ti,Density=InitN0,PIC=100 },
 --  H ={Type="DeltaF",Mass=mp,Charge=e,Temperature=Ti,Density=InitN0,PIC=100 }
 ele ={Type="DeltaF",Mass=me,Charge=-e,Temperature=Te,Density=InitN0,PIC=100 }
}
--]]


FieldSolver= 
{
--[[ 
  PML=  {Min={0.1*LX,0.1*LY,0.1*LZ},Max={0.9*LX,0.9*LY,0.9*LZ}}
 --]]
}

Constraints=
{
--[[
  { 
    DOF="E",IsHard=true,
	Select={Type="Boundary", Material="Vacuum" },
	Value= 0
  },

  { 
    DOF="J",
	Range={ {0.5*LX,0,0}},
	IsHard=true,
  	Value=function(t,x,y,z)	
         local tau = t*omega_ext  
         return { 0,math.sin(tau),0}   
      end
	 
 
  }
--]] 
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

-- The End ---------------------------------------

