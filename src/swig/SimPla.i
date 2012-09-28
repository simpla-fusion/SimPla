%module SimPla

%init %{
  import_array();
%}
#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include <std_shared_ptr.i> 

%include <std_string.i>
%include <std_pair.i>
%include <std_vector.i>
%include <typemaps.i>


%include "defs.h"
%include "exception.i"
%exception
{
  try
  {
    $action
  }
  catch (const std::invalid_argument& e)
  {
    SWIG_exception(SWIG_ValueError, e.what());
  }
  catch (const std::out_of_range& e)
  {
    SWIG_exception(SWIG_IndexError, e.what());
  }
}


%{
#include "engine/context.h"
%}


// typemap of nTuple =================================
%inline %{
#ifdef  SWIGPYTHON
#include "swig/pythonTypeMap.h"
#elif  SWIGLUA 
#include "swig/luaTypeMap.cpp"	 
#endif
%}


%typemap(in) IVec3 { toCXX($input,$1);} 

%typemap(out) IVec3 { $result=fromCXX($1);} 

%typemap(in)  Vec3 { toCXX($input,$1);} 

%typemap(out) Vec3 { $result=fromCXX($1);}  

%typemap(in)  boost::any{ toCXX($input,$1);}

%typemap(out) boost::any{ $result=fromCXX($1);}

%typemap(in)    std::map< std::string,boost::any > 
 { toCXX<  std::string, boost::any  >($input, $1 );}
 
%typemap(out)   std::map<std::string, boost::any >  
 { $result=fromCXX<   std::string, boost::any   >($1); }
 
 
%typemap(in) TR1::shared_ptr<ArrayObject>{ toCXX($input,$1);}

%typemap(out) TR1::shared_ptr<ArrayObject>{ $result=fromCXX($1);}
 
%typemap(in)  ArrayObject { toCXX($input,$1);}

%typemap(out)  ArrayObject { $result=fromCXX($1);} 
 
%shared_ptr(ArrayObject);
  
  
namespace std
{  
    %template(DVector)  std::vector<double>;
}

%shared_ptr(Grid);
%nodefaultctor Grid;
class Grid
{
public:
 static TR1::shared_ptr<Grid>  create(Real dt, Vec3 xmin, Vec3 xmax, IVec3 dims, IVec3 ghostWidth);
%immutable	;
	Real dt;
	// Geometry
    Vec3 xmin, xmax;
	Vec3 dx;
	Vec3 inv_dx;
	// Topology
	IVec3 dims;
	IVec3 ghostWidth;
	IVec3 strides;

%muttable;
};
    
%nodefaultctor Context;
%shared_ptr(Context);

class Context
{
public:


	typedef Context ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;
	
	 TR1::shared_ptr<Grid> grid;
	
 	static Holder create(std::string const &);
	
	std::string showSummary();
	static inline const char * showCopyright();
	static inline const char * showExampleConfig();
	
	void process();
	void pre_process();
	void post_process();

 	void addSpecies(std::string const & key, std::map<std::string, boost::any> property);
 	std::map<std::string, boost::any>  getSpecies(std::string const & key);
 	
	const std::string unit_dimensions;
	const Real Mu0; // H m^-1
	const Real Epsilon0; // F m^-1
	const Real SpeedOfLight; // m/s
	const Real ElementaryCharge; // C
	const Real ProtonMass; // kg
	const Real eV; //J

	inline SizeType getCounter() const;

	inline Real getTime() const;

    TR1::shared_ptr<ArrayObject> getFieldObject (const std::string & name,int  ,
    bool createIFNotExist=true );
	

};
 


