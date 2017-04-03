Config = {
    Atlas = {
        Dimensions = { 0, 0, 0 },
        Dx = { 1, 1, 1 },
        Origin = { 0, 0, 0 },
    },
    Domain = {
        Boundary = {
            Attributes = {
                B = {
                    name = "B",
                },
                E = {
                    name = "E",
                },
            },
            Mesh = "CartesianGeometry",
            Worker = {
                {
                    name = "PML",
                }
            },
            name = "Boundary",
        },
        Center = {
            Attributes = {
                B = {
                    SHARED = true,
                    name = "B",
                },
                E = {
                    SHARED = 1,
                    name = "E",
                },
                J1 = {
                    name = "J1",
                },
            },
            Mesh = "CartesianGeometry",
            Worker = {
                {
                    name = "EMFluid",
                }
            },
            name = "Center",
        },
    },
    Model = {
        Boundary = {
            GeoObject = { "+OuterBox", "-InnerBox" },
        },
        Center = {
            GeoObject = { "InnerBox" },
        },
    },
    name = "",
}

print(Config.Domain.Center.Worker[1].GUID)