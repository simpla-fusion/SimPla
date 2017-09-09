FILE(GLOB geometry_occ_SRC  *.cpp *.h  )
add_library(geometry_occ ${geometry_occ_SRC}  )

target_include_directories(geometry_occ BEFORE PRIVATE ${OCE_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(geometry_occ    utilities data ${OCE_LIBRARIES} )