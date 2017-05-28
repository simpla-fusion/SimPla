-- !/bin/lua
-- Created by IntelliJ IDEA.
-- User: salmon
-- Date: 17-5-28
-- Time: 下午6:06


Context = {
    Type = "MHDTest",
    Phi = { -3.14 / 8, 3.14 / 8 },
    Dimensions = { 32, 32, 32 },

    Domains =
    {

        Main={
            Type="HyperbolicConservationLawCartesianCoRectMesh"
        }

    },
    Model = {
--        MainBox={
--            Type="Cube",
--            lo={0,0,0},hi={1,1,1}
--        }
    },
    Atlas =    { PeriodicDimension = { 1, 1, 1 }, }
}

Schedule = {
    Type = "SAMRAITimeIntegrator",
    OutputURL = "MHDTestSaveData",
    TimeBegin = 0.0,
    TimeEnd = 5e-9,
    TimeStep = 1.0e-11,
    CheckPointInterval = 10
}


