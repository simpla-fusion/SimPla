Background  {#Background}
=============================================
 In the tokamak, from edge to core, the physical processes of plasma have different temporal-spatial
 scales and are described by different physical models. To achieve the device scale simulation, these
 physical models should be integrated into one simulation system. A reasonable solution is to reuse
 and couple existing software, i.e. integrated modeling projects IMI, IMFIT and TRANSP. However,
 different codes have different data structures and different interfaces, which make it a big challenge
 to efficiently integrate them together. Therefore, we consider another more aggressive solution,
 implementing and coupling different physical models and numerical algorithms on a unified framework
 with sharable data structures and software architecture. This is maybe more challenging, but can solve
 the problem by the roots.
 There are several important advantages to implement a unified software framework for the tokamak
 simulation system.
 - Different physical models could be tightly and efficiently coupled together. Data are shared in
   memory, and inter-process communications are minimized.
 - Decoupling and reusing physics independent functions, the implementation of new physical theory
   and model would be much easier.
 - Decoupling and reusing physics independent functions, the performance could be optimized by
   non-physicists, without any affection on the physical validity. Physicist   can easily take the
    benefit from the rapid growth of HPC.
 - All physical models and numerical algorithms applied into the simulation system could be comprehensively
   reviewed.
 To completely recover the physical process in the tokamak device, we need create a simulation system
  consisting of several different physical models. A unified development framework is necessary to
  achieve this goal.

\section  Detail  Detail
\f[
  |I_2|=\left| \int_{0}^T \psi(t)
           \left\{
              u(a,t)-
              \int_{\gamma(t)}^a
              \frac{d\theta}{k(\theta,t)}
              \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
           \right\} dt
        \right|
\f]

    \f$h\left(\psi,\theta\right)\f$  | \f$\theta\f$
	  	------------- | -------------
	  	\f$R/\left|\nabla \psi\right| \f$  | constant arc length
	  	\f$R^2\f$  | straight field lines
	  	\f$R\f$ | constant area
	     1  | constant volume




