SET(APPLICATION_SRC application.cpp application_main.cpp ../simulation/Context.cpp TimeDependentSolver.cpp )

add_library(task_flow ../simulation/Context.cpp ../simulation/ProblemDomain.cpp)
