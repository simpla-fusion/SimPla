simpla_test(engine_test engine_test.cpp
        ../../src/simpla/mesh/Mesh.cpp
        ../../src/simpla/mesh/StructuredMesh.cpp
        )
target_link_libraries(engine_test engine algebra  data utilities)

simpla_test(policies_test   policies_test.cpp)
