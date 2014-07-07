# Preliminary design {#PreliminaryDesign}

SimPla is a unified and hierarchical development framework for plasma simulation. Its long term goal
 is to provide complete modeling of a fusion device. “SimPla” is abbreviation of four words, Simulation,
  Integration, Multi-physics and Plasma.
The principle design ideas are explained in this section, and listed as:
* Idea(1) Ideas are expressed like this.

# TEst
# Design ideas
 
* Idea(2) SimPla is a unified framework.

“Unified” means all different physical models are implemented by this framework and tight coupled to
 each other under this framework. “Tight coupled” means physical models share data and data structure
 as much as possible, which is to reduce the overhead of data/data structure conversion and communication
 between models. SimPla is designed to couple and reuse physical models and numerical algorithms, but not
 to couple legacy codes. All physical models and some numerical algorithm need be recoded using this framework,
 before they are integrated into the simulation system. This decision bases on two reasons,
 
  1. Data conversion and communication between legacy codes are very complicate and will cause too much
 time lagged in the high performance simulation;
  2. The physical models and numerical algorithms used in the simulation system shall be comprehensive
reviewed and verified;

This decision is reasonable, but the cost for recoding all physical models will be very expensive. We
need reduce the development cost for the implement of physical model.

### Idea(3) SimPla is a hierarchical framework.

 “Hierarchical” means physical model, numerical algorithm and computing implement shall be decoupled.
The relationship between physical models and numerical algorithms is not 1:1. One physical equation
 may be solved by different numerical algorithm, and one numerical algorithm may be applied to different
 equation. The advantages of separating the numerical algorithm from physical model are listed as the follows:
1. Code reuse.
2. Data structure reuse.
3. Easy to implement one physical model by using different numerical algorithms. This made it easy to
 compare those different numerical algorithms, and is helpful to form a uniformed interface to the
 physical model (despite the used numerical algorithm)
4. Easy to implement new physical model by code and data structure reuse.
The separation of computation from numerical algorithms has similar advantages. And, in particular,
this separation makes it possible to perform the performance optimization on computational level without
affecting the physical model, numerical algorithms and program results.

* Idea(4) Discretization of partial differential equations using a human-readable syntax;

Usually, MHD theory is expressed by a group PDEs. Using C++ expression template technology, we can
 directly write these PDEs into the simulation code, and automatize the discretization process when
 the code is compiled. That means the code could be written in a physicist-friend style, and the
 development cost will be sharply decreased. The C++ expression template technology is realizable,
 which has a lot successful applications, i.e. OpenFOAM.




 
