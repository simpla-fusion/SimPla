--
-- User: salmon
-- Date: 11/5/15
-- Time: 3:53 PM
--

ProblemDomain = "EMFluid"

Description = "Cold Plasma Fluid Model" -- description or other text things.

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
epsilon0 = 8.8542e-12
--

k_parallel = 18
Btor = 2.0 * Tesla
Ti = 0.03 * KeV
Te = 0.05 * KeV
N0 = 1.0e19 -- m^-3


omega_ci = qe * Btor / mp -- e/m_p B0 rad/s
vTi = math.sqrt(k_B * Ti * 2 / mp)
rhoi = vTi / omega_ci -- m

omega_ce = qe * Btor / me -- e/m_p B0 rad/s
vTe = math.sqrt(k_B * Te * 2 / me)
rhoe = vTe / omega_ce -- m

omega_pe = math.sqrt(N0 * qe * qe / me / epsilon0)

NX = 50
NY = 50
NZ = 1
LX = 0.10 --m --100000*rhoi --0.6
LY = 0.10 --2.0*math.pi/k0
LZ = 1.0 -- math.pi * 0.25 -- 2.0*math.pi/18

number_of_steps = 20

step_of_check_point = 10


omega_ext = omega_pe * 0.8


Geometry = {
    Center = {
        Type = "Cube",
        Box = { { 0, 0, 0 }, { 1, 1, 1 } }
    }
}

MeshBase = {
    Type = "Cartesian",
    Origin = { 0, 0, 0 }, -- Dimensions[?]=1 => ignored dimension
    Dx = { 0.1, 0.1, 0.1 }, -- Dimensions[?]=1 => ignored dimension
    GhostWidth = { 2, 2, 2 }, -- GhostWidth[?]=0 => cycle boundary
}
Worker = { Type = "EMFluid" }




dt = 0.5 * (LX / NX) / c

PML = { Width = 50 }


InitValue = {
    B0 = {
        Value = function(x)
            return { 0, 0, 2 }
            --return { 0, 0, math.sin(x[1] * 2.0 * math.pi / LX) * math.sin(x[2] * 2.0 * math.pi / LY) }
        end
    },
    --    E1 = {
    --        Value = function(x)
    --            return { 0, 0, math.sin(x[1] * 2.0 * math.pi / LX) * math.sin(x[2] * 2.0 * math.pi / LY) }
    --        end
    --    },
}

R = function(x)
    return math.sqrt((x[1] - LX / 2.0) * (x[1] - LX / 2.0) + (x[2] - LY / 2.0) * (x[2] - LY / 2.0)) / (LX / 4.0)
end

--Particles = {
--    --    H = {
--    --        mass = mp,
--    --        charge = qe,
--    --        T = Ti,
--    --        pic = 2,
--    --        Type = "Boris",
--    --        IsParticle = true,
--    --        --  DisableCheckPoint = true,
--    --        DisableXDMFOutput = true,
--    --        V0 = { 1, 2, 3 },
--    --    },
--    ele = {
--        Mass = me,
--        Charge = -qe,
--        Type = "Fluid",
--        Box = { { 0, 0, 0 }, { LX, LY, LZ } },
--        Shape = function(x)
--            return R(x) - 1
--        end,
--        Density = function(x)
--            return N0 -- * (1.0 - math.cos((1.0 - R(x)) * math.pi * 0.5))
--        end
--    }
--}



Constraints = {
    E = {
        -- current source
        Box = { { 0.05 * LX, 0.45 * LY, 0.45 * LZ }, { 0.1 * LX, 0.55 * LY, 0.55 * LZ } },
        Value = function(x, t, v)
            local tau = t * omega_ext + x[1] * TWOPI / LX
            local amp = math.sin(tau) * (1 - math.exp(-tau * tau))
            return { 0, amp, 0 }
        end
    },
    --    PEC = {
    --        Bundle = {
    --            Box = { { 0, 0, 0 }, { LX, LY, 0 } },
    --            Shape = function(v)
    --                d1 = ((v[1] - LX / 2) * (v[1] - LX / 2) + (v[2] - LY / 2) * (v[2] - LY / 2)) - LY * LY * 0.04
    --                d2 = math.max(math.abs(v[1] - LX * 0.6) - 2, math.abs(v[2] - LY * 0.6) - 2)
    --                return math.min(d1, d2)
    --            end
    --        }
    --    }
}
