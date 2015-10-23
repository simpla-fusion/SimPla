Description = "For Cold Plasma Dispersion" -- description or other text things.

-- SI Unit System
c = 299792458 -- m/s
qe = 1.60217656e-19 -- C
me = 9.10938291e-31 --kg
mp = 1.672621777e-27 --kg
mp_me = 1836.15267245 --
KeV = 1.1604e7 -- K
Tesla = 1.0 -- Tesla
PI = 3.141592653589793
TWOPI = PI * 2
k_B = 1.3806488e-23 --Boltzmann_constant
--

k_parallel = 18
Btor = 1.0 * Tesla
Ti = 0.03 * KeV
Te = 0.05 * KeV
N0 = 1.0e17 -- m^-3


omega_ci = qe * Btor / mp -- e/m_p B0 rad/s
vTi = math.sqrt(k_B * Ti * 2 / mp)
rhoi = vTi / omega_ci -- m

omega_ce = qe * Btor / me -- e/m_p B0 rad/s
vTe = math.sqrt(k_B * Te * 2 / me)
rhoe = vTe / omega_ce -- m

NX = 64
NY = 64
NZ = 1
LX = 10 --m --100000*rhoi --0.6
LY = 10 --2.0*math.pi/k0
LZ = 1.0 -- 2.0*math.pi/18
GW = 5

Manifold =
{
    Geometry = {
        Topology = {
            Dimensions = { NX, NY, NZ },
        },
        Box = { { 0.0, 0.0, 0.0 }, { LX, LY, LZ } },
        dt = 0.5 * (LX / NX) / c
    }
}
omega_ext = 0.1 * math.pi / Manifold.Geometry.dt --omega_ci*1.9


--domain_center=function( x  )
--   return (x[0]-0.5)*( x[0]-0.5 ) +( x[1]-0.5 )*( x[1]-0.5 ) < 0.01
--end
domain_center = {
    --    Rectangle={{0.1,0.1,0},{0.2,0.2,0}} ,
    Polyline = {
        OnlyEdge = false,
        ZAXIS = 2,
        Points = { { 0.1, 0.1, 0 }, { 0.2, 0.2, 0 }, { 0.3, 0.4, 0 } }
    },
}

InitValue ={
    B = {
        Domain = { Box = { { 0, 0, 0 }, { LX, LY, LZ } } },
        Value = function(x, t)
            print(x[1],x[2],x[3],t)
            return { 0, 0, math.sin(x[1] * 2.0 * math.pi / LX) * math.sin(x[2] * 2.0 * math.pi / LY) }
        end
    },
    --  phi=
    --  {
    --    Domain={Box={{0 ,0 ,0},{LX,LY,LZ}}},
    --
    --    Value=function(x,t)
    --      -- print(x[1],x[2],x[3])
    --      return   math.sin(x[1]*0.92*math.pi)*math.sin(x[2]*0.02*math.pi)
    --    end
    --
    --  }
}

PEC = {
    Domain = {
        Box = { { 0, 0, 0 }, { LX, LY, 0 } },
        IsSurface = true,
        Object = function(v)
            d1 = ((v[1] - LX / 2) * (v[1] - LX / 2) + (v[2] - LY / 2) * (v[2] - LY / 2)) - LY * LY * 0.04
            d2 = math.max(math.abs(v[1] - LX * 0.6) - 2, math.abs(v[2] - LY * 0.6) - 2)
            --   print(v[1],v[2 ],v[3])
            return math.min(d1, d2)
        end
    }
}

Constraint = {
    J = {
        Domain = { Indices = { { NX / 4, NY / 4, 0, 2 } } },
        Value = function(x, t, v)
            local tau = t * omega_ext -- + x[2]*TWOPI/(xmax[3]-xmin[3])
            local amp = math.sin(tau) * (1 - math.exp(-tau * tau))

            return { 0, 0, amp }
        end
    },
}
