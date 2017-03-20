Config = {
    Atlas = {
        Dimensions = { 0, 0, 0 },
        Dx = { 1, 1, 1 },
        GUID = 16154369457276247333,
        Origin = { 0, 0, 0 },
    },
    DomainView = {
        Boundary = {
            Attributes = {
                B = {
                    GUID = 6076253175031252928,
                    name = "B",
                },
                E = {
                    GUID = 12023507626310113393,
                    name = "E",
                },
            },
            GUID = 4993983831546288099,
            Mesh = "CartesianGeometry",
            Worker = {
                {
                    GUID = 7982995043248908815,
                    name = "PML",
                }
            },
            name = "Boundary",
        },
        Center = {
            Attributes = {
                B = {
                    GUID = 10613602341015603317,
                    SHARED = true,
                    name = "B",
                },
                E = {
                    GUID = 17575601670236117662,
                    SHARED = 1,
                    name = "E",
                },
                J1 = {
                    GUID = 10228035809033728458,
                    name = "J1",
                },
            },
            GUID = 900322019422461586,
            Mesh = "CartesianGeometry",
            Worker = {
                {
                    GUID = 15708014333120542437,
                    name = "EMFluid",
                }
            },
            name = "Center",
        },
    },
    GUID = 6652661750295828350,
    Model = {
        Boundary = {
            GeoObject = { "+OuterBox", "-InnerBox" },
        },
        Center = {
            GeoObject = { "InnerBox" },
        },
        GUID = 11473906254360509429,
    },
    name = "",
}

print(Config.DomainView.Center.Worker[1].GUID)