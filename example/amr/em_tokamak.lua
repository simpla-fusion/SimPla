--
-- Created by IntelliJ IDEA.
-- User: salmon
-- Date: 17-4-27
-- Time: 上午10:40
-- To change this template use File | Settings | File Templates.
--

Context = {
    Type = "EMTokamak",
    gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900",
    Phi = { 0.0, 3.14/2},
    PeriodicDimension={1,0,1},
    Domains =
    {
        Center = { Type = "EMFluidCylindricalSMesh" },
      --   Limiter = { Type = "EMFluidCylindricalSMesh" },
      --  Atenna = { Type = "ICRF" }
    }
}

Schedule = {
    Type = "SAMRAITimeIntegrator"
}