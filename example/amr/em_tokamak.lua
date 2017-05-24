--! / bin / lua

Context = {
    Type = "EMTokamak",
    gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
    Phi = { -3.14 / 8, 3.14 / 8 },
    Dimensions = { 32, 32, 32 },
    Antenna = {
        x_lower = { 2.2, -0.1, -3.1415926 /64 },
        x_upper = { 2.25, 0.1, 3.1415926 / 64 },
        amp = { 0.0, 0.0, 1.0 },
        n_phi = 0.0,
        Frequency = 1.0e9
    },
    Domains =
    {
        Main = {
            Type = "EMFluidCylindricalSMesh",
--            Species = {
--                ele = { Z = -1.0, mass = 1.0 / 1836, ratio = 1.0 },
--                H = { Z = 1.0, mass = 1.0, ratio = 1.0 },
--            }
        },
    },
    Model = {},
    Atlas =
    {
        PeriodicDimension = { 0, 0, 0 },
    }
}

Schedule = { Type = "SAMRAITimeIntegrator", TimeBegin = 0.0, TimeEnd = 2e-9, TimeStep = 1.0e-11 }
