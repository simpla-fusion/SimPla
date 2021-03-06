--!/bin/lua
PI = 3.141592653589793
TWOPI = PI * 2.0
N_PHI = 100



Scenario = {
    Name = "EMTokamak",
    Mesh = {
        Type = "EBRectMesh",
        Chart =
        {
            Type = "Cylindrical",
            --            Origin = { 0.0, 0.0, 0.0 },
            --            Scale = { 1.0 / 64.0, 1.0 / 64.0, PI / 32 }, --Coarsest Cell Width
        },
        Box = {
            lo = { 1.2, -1.0, -PI / 4.0 },
            hi = { 2.5, 1.0, PI / 4.0 },
            Dimensions = { 64, 64, 64 }
        },
    },
    Model = {
        {
            Name = "Tokamak",
            Type = "Tokamak",
            gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
            Phi = { -PI / 4, PI / 4 },
        },
    },
    Domain = {
        --        Limiter = {
        --            Type = "Maxwell",
        --            --            Model = "Tokamak",
        --            --            Body = "Limiter",
        --        },
        --        PlasmaCenter = {
        --            Type = "EMFluid", -- "Domain<RectMesh,EBMesh,FVM,EMFluid>",
        --            Species = {
        --                ele = { Z = -1.0, mass = 1.0 / 1836, ratio = 1.0 },
        --                H = { Z = 1.0, mass = 1.0, ratio = 1.0 },
        --            },
        --            Model = "Tokamak",
        --            Body = "Plasma",
        --        },
        {
            Name = "ICRF",
            Type = "ICRFAntenna", -- "Domain<RectMesh,EBMesh,FVM,ICRFAntenna>",

            Boundary =
            {
                Type = "occ",
                File = "/home/salmon/workspace/SimPla/example/tokamak/coin.step",
                Location = { 1.8, -0.0, 0.1 },
                Scale = 1.0e-3
                --                Type = "Cube",
                --                lo = { 1.5, -0.5, -TWOPI / 8 },
                --                hi = { 2.0, 0.5, TWOPI / 8 },

                --                Type="Sphere",
                --                Origin={1.3,0,0 },
                --                Radius=0.1
            },
            IsHard = false,
            Amplify = { 0.0, 0.0, 1.0 },
            WaveNumber = { 0.0, 0.0, TWOPI / 12.0 },
            Frequency = 1.0e9,
        },
        {
            Name = "Plasma",
            Type = "PICBoris",
            Particle = {
                {
                    Name = "D",
                    M = 2.0,
                    Z = -1.0,
                    ratio = 1.0
                },
                {
                    Name = "H",
                    M = 1.0,
                    Z = 1.0,
                    ratio = 0.5
                },
            }
        }
    },
    Atlas = {
        PeriodicDimension = { 0, 0, 0 },
        SmallestPatchDimensions = { 8, 4, 8 },
        LargestPatchDimensions = { 64, 64, 64 },
        MaxLevel = 2;
    },
    Schedule = {
        Type = "SAMRAITimeIntegrator",
        OutputURL = "TokamakSaveData",
        TimeBegin = 0.0,
        TimeEnd = 5e-9,
        TimeStep = 1.0e-11,
        CheckPointInterval = 1,
        UpdateOrder = { "RFAntenna", "Tokamak" }
    }
}




