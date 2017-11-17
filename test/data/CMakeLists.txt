simpla_test(DataEntry_test DataEntry_test.cpp)
target_link_libraries(DataEntry_test
        -Wl,--whole-archive
          data  data_backend utilities
        -Wl,--no-whole-archive

        )
simpla_test(SPObject_test SPObject_test.cpp    )

target_link_libraries(SPObject_test
        -Wl,--whole-archive
        data data_backend
        -Wl,--no-whole-archive

        )
#SET(XML2_DIR ${PROJECT_SOURCE_DIR}/external_project/xdmf)
#SET(XDMF_DIR ${PROJECT_SOURCE_DIR}/external_project/xdmf)
#FIND_PACKAGE(XDMF)
#
#ADD_EXECUTABLE(XDMF_dummy   XDMF_dummy.cpp)
#target_include_directories(XDMF_dummy BEFORE PRIVATE   ${XDMF_INCLUDE_DIRS}  )
#target_link_libraries(XDMF_dummy  ${XDMF_LIBRARIES}  )