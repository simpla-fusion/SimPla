Description="For Cold Plasma Dispersion" -- description or other text things.    
-- SI Unit System
c = 299792458  -- m/s
KeV = 1.1604e7    -- K
Tesla = 1.0       -- Tesla    
--

Btor= 1.2  * Tesla
Ti =  0.03 * KeV
Te =  0.05 * KeV

rhoi = 4.57*1e-3 * math.sqrt(Ti)/Btor  --1.02 * math.sqrt(Ti)/(1e4*Btor) -- m
--    print(rhoi)

k0 = 25./40.
NX = 101
NY = 1
NZ = 1
LX = 100 --0.6
LY = 0 --2.0*math.pi/k0
LZ = 0  -- 2.0*math.pi/18
GW = 5 
N0 = 0.8*0.25e18 -- 4*Btor*Btor* 5.327934360e15 -- m^-3

omega_ci = 9.578309e7 * Btor -- e/m_p B0 rad/s

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
      return {Btor,0,0}  
     end
      ,
  E0=0.0, E1=0.0, B1=0.0,J1=0.0
}
-- GFile
Grid=
{
  Type="CoRectMesh",
  ScalarType="Complex",
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
--      dt= 2.0*math.pi/omega_ci/100.0
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

MediaTag=
{
  DefaultTag="Vacuum",
  {Region={{0.05*LX,0.0,0.0},{LX,0.0,0.0}},Type="Plasma"}
}
Interface
{
   {In="Plasma",Out="Vacuum",Type="PEC"},
   {In="Vacuum",Out="NONE",Type="PML"} 
}

CurrentSrc=
 { 
  Points={{LX/2.0,0.0,0.0},},
  Fun=function(x,y,z,t)
        local tau = t*omega_ci
--        print(tau)
      return {0,math.sin(tau)*(1-math.exp(-tau*tau)),0}   
      end
 }


-- The End ---------------------------------------

