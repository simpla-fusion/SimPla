/*
 * context.cpp
 *
 *  Created on: 2011-12-12
 *      Author: salmon
 */

#include "context.h"
namespace simpla
{

const char Context::Copyright[] = "\nSimPla aka. GGauge, 3D version \n"
		"Electromagnetic Plasma Kinetics and Fluid Simulation"
		"\nBuild: " __DATE__ " " __TIME__" "
#ifdef IDENTIFY
		" Rev. ID: " IDENTIFY " "
#endif
		"\nCopyright (C) 2007-2011 YU Zhi. ASIPP.  All rights reserved.";

const char Context::ExampleConfigFile[] =

		"-- Auto-generated  config file.------------------- --------\n"
				"-- "__DATE__ " " __TIME__ " \n"
		"    UNIT_DIMENSIONS=\"SI\"    \n"
		"             \n"
		"    DESCRIPTION=\"bala bala bong\" -- description or other text things.    \n"
		"             \n"
		"    DIMS={107,107,1}  -- number of grid, now only first dimension is valid       \n"
		"             \n"
		"    GW= {3,3,3}  -- width of ghost points            \n"
		"             \n"
		"    LENGTH={20*math.pi/(DIMS[1]-GW[1]*2-1)*(DIMS[1]-1),    \n"
		"        20*math.pi/(DIMS[2]-GW[2]*2-1)*(DIMS[2]-1),    \n"
		"        20*math.pi/(DIMS[3]-GW[3]*2-1)*(DIMS[3]-1)}  "
		"-- length of simulation domain     \n"
		"             \n"
		"    DT=0.5*LENGTH[1]/(DIMS[1]-1)    -- time step      \n"
		"             \n"
		"    DIAGNOSIS={\"E1\",\"B1\",}           \n"
		"             \n"
		"    SP_LIST={\"H_g\",\"ele\"}  "
		"-- the list of species in the simulation        \n"
		"             \n"
		"    SPECIES= -- predefine species         \n"
		"    {                \n"
		"         ele={desc=\"ele\" ,Z=-1.0,  "
		"  m=1.0/1836.0, ns=1.0,  engine=\"ColdFulid\"},   \n"
		"             \n"
		"         H_c={desc=\"H\"    ,Z=1.0,  "
		"   m=1,    ns=1.0,     engine=\"ColdFulid\"},       \n"
		"             \n"
		"         H_g={desc=\"H\"    ,Z=1.0,   m=1,    ns=1.0,           \n"
		"             xmin={0,0,0},xmax={LX,LY,LZ},engine=\"GyroGauge\",   "
		"  T=1.0e-4, numOfMate=20,    PIC=10},   \n"
		"    }         \n"
		"    B0={}  -- initial/background magnetic field   \n"
		"    E0={}  -- initial/background electric field; "
		"is zero if E0==null        \n"
		"    N0={}  -- initial/background"
		" density field of  electron;         \n"
		"\n"
		"    for x=0,DIMS[1] -1 do        \n"
		"     for y=0,DIMS[2] -1 do     \n"
		"      for z=0,DIMS[3] -1 do    \n"
		"        s=x*DIMS[2]*DIMS[3]+ y*DIMS[3]+z    \n"
		"        N0[s] = 1.0     \n"
		"         B0[s*3+0]=0.0    \n"
		"         B0[s*3+1]=0.0     \n"
		"         B0[s*3+2]=1.0    \n"
		"      end    \n"
		"     end    \n"
		"    end     \n"
		"\n"
		"    ---[[ uncomment this line, if you need Cycle BC.\n"
		"    -- set BC(boundary condition), now only first two "
		"are valid         \n"
		"    -- BC >= GW               \n"
		"    BC= {             \n"
		"        -1,-1, -- direction x  \n"
		"        -1,-1, -- direction y  \n"
		"        -1,-1 -- direction z  \n"
		"        }            \n"
		"    for x=0,DIMS[1] -1 do         \n"
		"     for y=0,DIMS[2] -1 do    \n"
		"      for z=0,DIMS[3] -1 do     \n"
		"        s=x*DIMS[2]*DIMS[3]+ y*DIMS[3]+z    \n"
		"        qx = (x-GW[1])%(DIMS[1]-2*GW[1]-1)"
		"/(DIMS[1]-2*GW[1]-1) *2.0 *math.pi    \n"
		"        qy = (y-GW[2])%(DIMS[2]-2*GW[2]-1)"
		"/(DIMS[2]-2*GW[2]-1) *2.0 *math.pi    \n"
		"        qz = (z-GW[3])%(DIMS[3]-2*GW[3]-1)"
		"/(DIMS[3]-2*GW[3]-1) *2.0 *math.pi    \n"
		"        a=0.0    \n"
		"        for kx=1,1 do    \n"
		"         for ky=1,1 do    \n"
		"            a=a+math.sin(qx*kx)*math.sin(qy*ky)    \n"
		"         end    \n"
		"        end    \n"
		"    \n"
		"        E0[s*3+0]=0.0     \n"
		"        E0[s*3+1]=1.0e-8 *a    \n"
		"        E0[s*3+2]=1.0e-8 *a    \n"
		"      end    \n"
		"     end    \n"
		"    end    \n"
		"    --]]            \n"
		"             \n"
		"    --[[ uncomment this line, if you need absorbing condition. \n"
		"    BC= {            \n"
		"         1,1, -- direction x \n"
		"         -1,-1, -- direction y \n"
		"         -1,-1 -- direction z \n"
		"         }            \n"
		"    SrcPos=DIMS[1]/2*DIMS[2]*DIMS[3] "
		"+DIMS[2]/2*DIMS[3] + DIMS[3]/2               \n"
		"    Srcf0=1.9               \n"
		"    Srckx=0            \n"
		"    Srctau=0.5               \n"
		"             \n"
		"    function JSrc(t)              \n"
		"         alpha=t*Srcf0 - Srctau*math.pi*2.0        \n"
		"         res={}               \n"
		"         if alpha<0 then              \n"
		"               a=math.sin(alpha)*math.exp(-alpha*alpha)       \n"
		"         else               \n"
		"           a=math.sin(alpha)          \n"
		"         end            \n"
		"         res[SrcPos*3+0]= 0.0             \n"
		"         res[SrcPos*3+1]= 1.0e-8*a         \n"
		"         res[SrcPos*3+2]= 0.0             \n"
		"         return res           \n"
		"    end                \n"
		"    --]]            \n"
		"-- The End ---------------------------------------\n";

Context::Context(std::string const & u) :
		counter_(0), time_(0), unit_dimensions(u),

		Mu0((u == "SI") ? CONST_SI_PERMEABILITY : 1),

		Epsilon0((u == "SI") ? CONST_SI_PERMITTIVITY : 1),

		SpeedOfLight((u == "SI") ? CONST_SI_C : 1),

		ElementaryCharge((u == "SI") ? CONST_SI_ELEMENT_CHARAGE : 1),

		ProtonMass((u == "SI") ? CONST_SI_PROTON_MASS : 1),

		eV((u == "SI") ? ElementaryCharge : 1)
{
}
Context::~Context()
{
}

std::string Context::ShowSummary()
{

	std::cout << SINGLELINE << std::endl;

	std::cout << "[ Summary of Environment ]" << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << "[Description]" << std::endl;

	std::cout << desc << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << std::setw(20) << "Building Time : " << (__TIME__) << " "
			<< (__DATE__) << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << "[Unit & Dimensions]" << std::endl;

	std::cout << std::setw(20) << "Unit Dimensions : " << unit_dimensions
			<< std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << "[Spatial Grid/TWOPI]" << std::endl;

	std::cout << std::setw(20) << "Grid dims : " << grid.dims << std::endl;

	std::cout << std::setw(20) << "Range : " << grid.xmin << " ~ " << grid.xmax
			<< "[m]" << std::endl;

	std::cout << std::setw(20) << "dx : " << grid.dx << "[m]" << std::endl;

	std::cout << std::setw(20) << "dt : " << grid.dt << "[s]" << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << "[Boundary Condition]" << std::endl
			<< " (< 0 for CYCLE, 0 for PEC, 1 for Mur ABC ,>1 for PML ABC)"
			<< std::endl;

	std::cout << std::setw(20) << "LEFT : " << bc[0] << std::endl;

	std::cout << std::setw(20) << "RIGHT : " << bc[1] << std::endl;

	std::cout << std::setw(20) << "FRONT : " << bc[2] << std::endl;

	std::cout << std::setw(20) << "BACK : " << bc[3] << std::endl;

	std::cout << std::setw(20) << "UP : " << bc[4] << std::endl;

	std::cout << std::setw(20) << "DOWN : " << bc[5] << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << "[ Functions List]" << std::endl;

	std::cout << SINGLELINE << std::endl;

	std::cout << std::endl //
			<< SINGLELINE << std::endl //
			<< "[Predefine Species]" << std::endl //
			<< SINGLELINE << std::endl //
			<< std::setw(20) << " Name  | " //
			<< " Description" << std::endl;
	for (Context::SpeciesMap::iterator it = species_.begin();
			it != species_.end(); ++it)
	{
		std::cout

		<< std::setw(17) << it->first << " | "

		<< " q/e = " << boost::any_cast<double>(it->second["Z"])

		<< ", m/m_p = " << boost::any_cast<double>(it->second["m"])

		<< ", T = " << boost::any_cast<double>(it->second["T"]) << "[eV]"

		<< ", pic = " << boost::any_cast<double>(it->second["pic"])

		<< std::endl

		<< std::setw(20) << " | " << ", engine = "

		<< boost::any_cast<std::string>(it->second["engine"])

		<< std::endl;
	}
	std::cout << SINGLELINE << std::endl;

	return ("");

}

} // namespace simpla
