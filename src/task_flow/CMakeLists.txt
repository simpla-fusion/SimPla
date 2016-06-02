SET(APPLICATION_SRC application.cpp application_main.cpp Context.cpp TimeDependentSolver.cpp )

add_library(task_flow Context.cpp ProblemDomain.cpp)
