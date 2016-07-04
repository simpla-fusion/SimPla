--
-- User: salmon
-- Date: 11/5/15
-- Time: 3:53 PM
--

ProblemDomain = "PIC" --"Fluid"

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
Btor = 2.0 * Tesla
Ti = 0.03 * KeV
Te = 0.05 * KeV
N0 = 1.0e17 -- m^-3


omega_ci = 10 * qe * Btor / mp -- e/m_p B0 rad/s
vTi = math.sqrt(k_B * Ti * 2 / mp)
rhoi = vTi / omega_ci -- m

omega_ce = qe * Btor / me -- e/m_p B0 rad/s
vTe = math.sqrt(k_B * Te * 2 / me)
rhoe = vTe / omega_ce -- m

NX = 50
NY = 50
NZ = 1
LX = 1.0 --m --100000*rhoi --0.6
LY = 2.0 --2.0*math.pi/k0
LZ = 3.0 -- math.pi * 0.25 -- 2.0*math.pi/18
GW = 5
PIC = 100
GEQDSK = "/home/salmon/workspace-local/SimPla/scripts/gfile/g038300.03900"
number_of_steps = 1
dt = 0.5 * (LX / NX) / c
--current_time = 0;
stop_time = dt * number_of_steps;
step_of_check_point = 1
Mesh =
{
    Dimensions = { NX, NY, NZ },
    Box = { { 0.0, 0.0, 0 }, { LX, LY, LZ } },
    dt = 0.5 * (LX / NX) / c
}
omega_ext = omega_ci * 1.9


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

InitValue = {
    B0 = {
        Value = function(x)
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
    --    E1 = {
    --        Value = function(x)
    --            local tau = x[1] * TWOPI / LX + x[2] * TWOPI / LY + x[3] * TWOPI / LZ
    --
    --            return {
    --                math.sin(tau),
    --                math.sin(tau + TWOPI / 3.0),
    --                math.sin(tau + TWOPI * 2.0 / 3.0)
    --            }
    --        end
    --    },
}
Particles = {
    H = {
        mass = mp,
        charge = qe,
        T = Ti,
        pic = 2,
        Type = "Boris",
        IsParticle = true,
        --  DisableCheckPoint = true,
        DisableXDMFOutput = true,
        V0 = { 1, 2, 3 },
    },
    ele = {
        mass = me,
        charge = -qe,
        T = Te,
        pic = PIC,
        --        Density = function(t, x)
        --            return (1.0 - math.cos(x[1] / LX * math.pi * 2.0)) / 2 / PIC
        --        end
    }
}

PML = { Width = 20 }

PEC = {
    Domain = {
        Box = { { 0, 0, 0 }, { LX, LY, 0 } },
        IsSurface = true,
        Object = function(v)
            d1 = ((v[1] - LX / 2) * (v[1] - LX / 2) + (v[2] - LY / 2) * (v[2] - LY / 2)) - LY * LY * 0.04
            d2 = math.max(math.abs(v[1] - LX * 0.6) - 2, math.abs(v[2] - LY * 0.6) - 2)
            -- print(v[1], v[2], v[3])
            return math.min(d1, d2)
        end
    }
}

Constraints = {
    J = {
        Box = { { 0.45 * LX, 0.45 * LY, 0.45 * LZ }, { 0.55 * LX, 0.55 * LY, 0.55 * LZ } },
        Value = function(t, x, v)
            local tau = t * omega_ext + x[1] * TWOPI / LX
            local amp = math.sin(tau) * (1 - math.exp(-tau * tau))
            return { 0, 0, amp }
        end
    },
}
