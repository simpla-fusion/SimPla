#simpla_test(kinetic_particle_test kinetic_particle_test.cpp)
#target_link_libraries(kinetic_particle_test physics io parallel utilities data_model)
#
#simpla_test(probe_particle_test probe_particle_test.cpp)
#target_link_libraries(probe_particle_test physics io parallel utilities)
#
ADD_EXECUTABLE(SmallObjPoolC_test SmallObjPoolC_test.c)
target_link_libraries(SmallObjPoolC_test sp)

ADD_EXECUTABLE(SmallObjPoolCXX_test SmallObjPoolCXX_test.cpp)
target_link_libraries(SmallObjPoolCXX_test sp)
