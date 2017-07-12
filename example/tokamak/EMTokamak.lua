--! / bin / lua
PI = 3.141592653589793
TWOPI = PI * 2.0
N_PHI = 100



Context = {
    Name = "EMTokamak",
    Atlas = {
        Chart = { Type = "Cylindrical" },
        IndexOrigin = { 0, 0, 0 },
        Dimensions = { 32, 32, 32 },
        PeriodicDimension = { 0, 0, 1 },
        lo = { 1.4, -1.0, -PI / 2 },
        hi = { 2.5, 1.0, PI / 2 }
    },
    Domain =
    {
        Tokamak = {
            Type = "Domain<RectMesh,FVM,EMFluid>",
            Species = {
                ele = { Z = -1.0, mass = 1.0 / 1836, ratio = 1.0 },
                H = { Z = 1.0, mass = 1.0, ratio = 1.0 },
            },
            Model = {
                Type = "Tokamak",
                gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
                Phi = { -TWOPI / 4, TWOPI / 4 },
            },
        },
        RFAntenna = {
            Type = "Domain<RectMesh,FVM,ICRFAntenna>",
            Variable = { Name = "E" },
            IsHard = false,
            Amplify = { 0.0, 0.0, 1.0 },
            WaveNumber = { 0.0, 0.0, TWOPI / 12.0 },
            Frequency = 1.0e9,
            Model = {
                GeoObects = {
                    Boundary = {
                        Type = "Cube",
                        lo = { 2.2, -0.1, -TWOPI / 8 },
                        hi = { 2.25, 0.1, TWOPI / 8 }
                    }
                }
            },
        },
    },
}

Schedule = {
    Type = "SAMRAITimeIntegrator",
    OutputURL = "TokamakSaveData",
    TimeBegin = 0.0,
    TimeEnd = 5e-9,
    TimeStep = 1.0e-11,
    CheckPointInterval = 1,
    UpdateOrder = { "RFAntenna", "Tokamak" }
}
