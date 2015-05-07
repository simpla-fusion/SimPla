Description="  Plasma PIC" -- description or other text things.

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

NX = 100
NY = 1
NZ = 1
LX = 1  --m --100000*rhoi --0.6
LY = 2  --2.0*math.pi/k0
LZ = 3  -- 2.0*math.pi/18
GW = 5

PIC=200

omega_ext=omega_ci*1.9

Mesh=
  {
    Dimensions={NX,NY,NZ}, -- number of grid, now only first dimension is valid

    Box={ {0.0, 0.0 , 0.0} , {LX,LY,LZ} },

    dt=0.5*LX/NX/c  -- time step
  }


Particle=
  {
    H={
      mass=1.0,charge=2.0,T=3.0,pic=PIC,

      Distribution=
      function(x,v)
        return (1.0-math.cos(x[1]/LX*math.pi*2.0))/2/PIC
      end

    }
  }
