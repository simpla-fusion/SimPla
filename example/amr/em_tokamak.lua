--
-- Created by IntelliJ IDEA.
-- User: salmon
-- Date: 17-4-27
-- Time: 上午10:40
-- To change this template use File | Settings | File Templates.
--

Context = {
    Type = "EMTokamak",
    Model = {
        tokamak = { Type = "GEqdsk", gfile = "/home/salmon/workspace/SimPla/scripts/gfile/g038300.03900" }
    },
    Domains =
    {
        Center = { Type = "EMFluid" },
        Limter = { Type = "PEC" },
        Atenna = { Type = "ICRF" }
    }
}

Schedule = {
    Type = "SAMRAITimeIntegrator"
}