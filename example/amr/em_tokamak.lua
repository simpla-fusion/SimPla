-- !/bin/lua

Context = {
    Type = "EMTokamak",
    gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
    Phi = { -3.14/4, 3.14/4 },
    Dimensions = { 32, 32, 8 },
    Antenna = {
        x_lower = { 1.7, -0.2,  -3.1415926/16 },
        x_upper = { 2.0, 0.2,   3.1415926/16  },
        amp = {0.0,0.0,1.0} ,
        n_phi = 1.0,
        Frequency = 1.0e9
    },
    Domains =
    {
        Main = { Type = "EMFluidCylindricalSMesh" },
    },
    Model =
    {},
    Atlas = { PeriodicDimension = { 0, 1, 1 }, }
}

Schedule = {
    Type = "SAMRAITimeIntegrator",
    TimeBegin=0.0,
    TimeEnd=10.0e-8,
    TimeStep=1.0e-9
}
