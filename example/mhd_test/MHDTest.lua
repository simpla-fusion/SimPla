-- !/bin/lua
-- Created by IntelliJ IDEA.
-- User: salmon
-- Date: 17-5-28
-- Time: 下午6:06


TWOPI = 3.141592653589793 * 2.0
N_PHI = 100
Context = {
    Name ="MHDTest",
    Mesh = {
        Type = "RectMesh",
        Coordinates = { Type = "Cylindrical" },
        -- IndexOrigin = { 0, 0, 0 },
        Dimensions = { 32, 32, 32 },
        PeriodicDimension = { 1, 1, 1 },
    },
    Model = {
        Box = {
            Type = "Cube",
            lo = { 2.0, -2.0, -TWOPI / 8 },
            hi = { 4.0, 2.0, TWOPI / 8 }
        },
    },
    Domain =
    {
        MHDTest = {
            Type = "IdealMHD",
        },
    },
}

Schedule = {
    Type = "SAMRAITimeIntegrator",
    OutPutPrefix= "",
    TimeBegin = 0.0,
    TimeEnd = 5e-9,
    TimeStep = 1.0e-11,
    CheckPointInterval = 1,
    UpdateOrder = { "RFAntenna", "Tokamak" }
}
