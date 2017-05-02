-- !/bin/lua

Context = {
    Type = "EMTokamak",
    gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
    Phi = { 0.0, 3.14 / 2 },
    Domains =
    {
        Center = { Type = "EMFluidCylindricalSMesh" },
        Boundary = { Type = "EMFluidCylindricalSMesh" },
        --   Atenna = { Type = "ICRF" ,
        --         WaveNumber={0,0,1},
        --         Frequence=1.9e19,
        --         Amplify= 1.0e5,
        --         },
    },
    Model =
    {--      Atenna={ Type="NIL"}
    },
    Atlas = { PeriodicDimension = { 0, 0, 1 }, }
}

Schedule = {
    Type = "SAMRAITimeIntegrator"
}
