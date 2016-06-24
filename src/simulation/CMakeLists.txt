SET(APPLICATION_SRC application.cpp application_main.cpp Context.cpp TimeDependentSolver.cpp)

add_library(simulation Context.cpp ProblemDomain.cpp)
