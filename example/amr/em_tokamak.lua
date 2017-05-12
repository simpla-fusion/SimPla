-- !/bin/lua

Context = {
    Type = "EMTokamak",
    gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
    Phi = { 0.0, 3.14*2 },
    Dimensions = { 64, 64, 32 },
    Antenna = {
        x_lower = { 1.4, -0.1, 0 },
        x_upper = { 1.45, 0.1, 3.1415926/2  },
        amp = 1.0,
        n_phi = 1.0,
        Frequency = 1.0e8
    },
    Domains =
    {
        Limiter = { Type = "EMFluidCylindricalSMesh" },
        --   Atenna = { Type = "ICRF" ,
        --         WaveNumber={0,0,1},
        --         Frequence=1.9e9,
        --         Amplify= 1.0e5,
        --         },
    },
    Model =
    {},
    Atlas = { PeriodicDimension = { 0, 0, 1 }, }
}

Schedule = {
    Type = "SAMRAITimeIntegrator",
    TimeBegin=0.0,
    TimeEnd=4.0e-8,
    TimeStep=1.0e-9
}
