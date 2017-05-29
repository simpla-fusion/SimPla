--! / bin / lua

Context = {
    CoordinateSystem = {
        BigCylindrical = {
            Type = "Cylindrical",
            PeriodicDimension = { 0, 0, 1 },
            dx = { 0.1, 0.1, 0.1 }
        },
    },
    Domains =
    {
        Tokamak = {
            Type = "EMFluid",
            CoordinateSystem = "BigCylindrical",
            Mesh = "SMesh",
            Model = "Tokamak",
            BoundaryCondition = {
                Type = "PEC",
                Model = "Tokamak.Limiter",
            },
            InitialCondition = {
                ne = "Tokamak.ne",
                B0v = "Tokamak.B0",
            },
            Species = {
                ele = { Z = -1.0, mass = 1.0 / 1836, ratio = 1.0 },
                H = { Z = 1.0, mass = 1.0, ratio = 1.0 },
            }
        },
        Antenna = {
            Type = "ExtraSource",
            CoordinateSystem = "BigCylindrical",
            Mesh = "SMesh",
            Variable = "E",
            IsHard = false,
            Model = "Antenna",
            Amplify = { 0.0, 0.0, 1.0 },
            WaveNumber = 0.0,
            Frequency = 1.0e9
        },
    },
    Model = {
        Tokamak = {
            Type = "Tokamak",
            CoordinateSystem = "BigCylindrical",
            gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
            Phi = { -3.14 / 8, 3.14 / 8 },
        },
        Antenna = {
            Type = "Cube",
            CoordinateSystem = "BigCylindrical",
            lo = { 2.2, -0.1, -3.1415926 / 64 },
            hi = { 2.25, 0.1, 3.1415926 / 64 }
        },
    },
}

Schedule = {
    Backend = "SAMRAITimeIntegrator",
    OutputURL = "TokamakSaveData",
    TimeBegin = 0.0,
    TimeEnd = 5e-9,
    TimeStep = 1.0e-11,
    CheckPointInterval = 10,
    UpdateOrder = { "Antenna", "Core" }
}
