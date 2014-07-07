# Requirement {#Requirement}
 
The final objective of SimPla is to provide a complete and efficient framework for fusion plasma
 simulation. A list of strategic requirements has been established. They are explained in this section
 , and listed as:
* Req(1) Requirements are expressed like this.

1.1 Physics model and numerical algorithm
-----------------------

 In the tokamak, from edge to core, the physical processes of plasma have different temporal-spatial
scales and are described by different physical models.
The physical properties of simulation system are described by physical quantities. Physical quantity
 could be a spatial field, i.e. electric field, magnetic field, density, current, etc. or a phase space
 distribution function. The physical theory about the relation between different physical quantities
 and the temporal evolution of physical quantity are referred to as physical model or physical laws.
 Physical models are usually expressed by group partial differential equations (PDE), i.e. Maxwell
 equations, MHD equations or Vlasov Equations etc. The number of equations in the group should be same
  as the number of unknown physical quantities in the domain.  One physical quantity may follow different
   physical models in different spatial domains. At the boundary of adjacent domains, physical models
   are coupled to each other through their common physical quantities.
The physical equations may be expressed on different coordinates systems, i.e. Cartesian coordinates,
cylindrical coordinates, toroidal coordinates and magnetic flus coordinates etc.
To numerically solve physical equations, the physical quantities are approximately represented by
values on discrete space points (mesh), and equations of continuous quantities are approximately
converted into algebra equations of discrete values.  The method to construct this discrete approximation
 is referred to as numerical algorithms, i.e. finite difference method (FDM), finite element method (FEM),
 finite volume method (FVM), Particle-in-cell method (PIC) etc. One physical equation may be solved by
  different numerical algorithm, and one numerical algorithm may be applied to different equation.

* Req(2) Supports different physical model, i.e. Maxwell equations, MHD equations or Vlasov Equations etc.
* Req(3) Supports different coordinates system, i.e. Cartesian coordinates, cylindrical coordinates,
toroidal coordinates and magnetic flus coordinates etc.
* Req(4) Supports different numerical algorithm, i.e. FDM, FVM, FEM, DG-FEM, PIC, Delta-f etc.
* Req(5) Automatize or simplify the conversion from physical equation to numerical algorithm;

1.2 Flexible and verifiability
-----------------------

* Req(6) Physical model and numerical algorithm can be verified independently.
* Req(7) Be flexible to efficiently solve different scale problems, from one-dimensional slab model to
three-dimensional global toroidal model.

1.3 Module and Integration
-----------------------
* Req(8) Different physical model and numerical algorithms shall be coupled in memory;
* Req(9) Each module can be independent validated;
* Req(10) Support flexible task flow control;
* Req(11) Support global (device-scale) multi-module integration;


1.4 High performance computing
-----------------------
* Req(12) Supports large scale high performance computing;
* Req(13) Has good portability on different hardware architectures, supports main-stream high performance
hardware architectures, i.e. multi-core CPU, many-core GPU;


1.5 Input/output
-----------------------
* Req(14) Provide interface for external pre-process software, i.e. modeling and mesh generate tools;
* Req(15) Provide interface for external post-process software, i.e. visualization tools
* Req(16) Flexible configuration;
* Req(17) Comprehensive and expressive logging information;