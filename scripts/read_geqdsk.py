#!/usr/bin/env python
import numpy as np
    
def read_geqdsk(filename):
    """
    'desc'    :desc,  
    'nw'      :nw,    # Numver of horizontal R grid  points
    'nh'      :nh,    # Number of vertical Z grid points
    'rdim'    :rdim,  # Horizontal dimension in meter of computational box
    'zdim'    :zdim,  # Vertical dimension in meter of computational box
    'rcentr'  :rcentr,# 
    'rleft'   :rleft, # Minimum R in meter of rectangular computational box
    'zmid'    :zmid,  # Z of center of computational box in meter
    'rmaxis'  :rmaxis,# R of magnetic axis in meter
    'rmaxis'  :zmaxis,# Z of magnetic axis in meter
    'simag'   :simag, # poloidal flus ax magnetic axis in Weber / rad
    'sibry'   :sibry, # Poloidal flux at the plasma boundary in Weber / rad
    'rcentr'  :rcentr,# R in meter of  vacuum toroidal magnetic field BCENTR
    'bcentr'  :bcentr,# Vacuum toroidal magnetic field in Tesla at RCENTR
    'current' :current,# Plasma current in Ampere 
    'fpol'    :fpol,  # Poloidal current function in m-T, $F=RB_T$ on flux grid
    'pres'    :pres,  # Plasma pressure in $nt/m^2$ on uniform flux grid
    'ffprim'  :ffprim,# $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
    'pprim'   :pprim, # $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid
    'psizr'   :psizr, # Poloidal flus in Webber/rad on the rectangular grid points
    'qpsi'    :qpsi,  # q values on uniform flux grid from axis to boundary
    'nbbbs'   :nbbbs, # Number of boundary points
    'limitr'  :limitr,# Number of limiter points
    'rbbbs'   :rbbbs, # R of boundary points in meter
    'zbbbs'   :zbbbs, # Z of boundary points in meter
    'rlim'    :rlim,  # R of surrounding limiter contour in meter
    'rlim'    :zlim,  # R of surrounding limiter contour in meter
    
    Toroidal Current Density
     $J_T(Amp/m^2)= R P^\prim(\psi)+ F F^\prim(\psi)/R/\mu_0$
    """
    d=open(filename,"r").read().replace("\n","");
    desc	= d[0:48]
   # idum	= int(d[48:52])
    nw	     = int(d[52:56])
    nh	     = int(d[56:60])
    it=60
    (rdim,zdim,rcentr,rleft,zmid,
    rmaxis,zmaxis,simag,sibry,bcentr,
    current,simag,xdum,rmaxis,xdum,
    zmaxis,xdum,sibry,xdum,xdum)=(float(d[it+i*16:it+(i+1)*16]) for i in range(20))
    it+=20*16;
    
    fpol=np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw)]);
    it+=nw*16;
    
    pres=np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw)]);
    it+=nw*16;
    
    ffprim=np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw)]);
    it+=nw*16;
    
    pprime=np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw)]);
    it+=nw*16;
    
    psirz=np.reshape(np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw*nh)]),(nw,nh));
    it+=nh*nw*16;
    
    qpsi=np.array([float(d[it+i*16:it+(i+1)*16]) for i in range(nw)]);
    it+=nw*16;
    
    nbbbs=int(d[it:it+5])
    limitr=int(d[it+5:it+10])
    it+=10
    
    rbbbs=np.array([float(d[it+i*32:it+i*32+16]) for i in range(nbbbs)]);
    zbbbs=np.array([float(d[it+i*32+16:it+i*32+32]) for i in range(nbbbs)]);
    it+=nbbbs*16*2;
    
    rlim=np.array([float(d[it+i*32:it+i*32+16]) for i in range(limitr)]);
    zlim=np.array([float(d[it+i*32+16:it+i*32+32]) for i in range(limitr)]);
    
    
    return {
    'desc'    :desc,  
    'nw'      :nw,    # Number of horizontal R grid  points
    'nh'      :nh,    # Number of vertical Z grid points
    'rdim'    :rdim,  # Horizontal dimension in meter of computational box
    'zdim'    :zdim,  # Vertical dimension in meter of computational box
    'rcentr'  :rcentr,# 
    'rleft'   :rleft, # Minimum R in meter of rectangular computational box
    'zmid'    :zmid,  # Z of center of computational box in meter
    'rmaxis'  :rmaxis,# R of magnetic axis in meter
    'zmaxis'  :zmaxis,# Z of magnetic axis in meter
    'simag'   :simag, # poloidal flus ax magnetic axis in Weber / rad
    'sibry'   :sibry, # Poloidal flux at the plasma boundary in Weber / rad
    'rcentr'  :rcentr,# R in meter of  vacuum toroidal magnetic field BCENTR
    'bcentr'  :bcentr,# Vacuum toroidal magnetic field in Tesla at RCENTR 
    'current' :current,# Plasma current in Ampere
    'fpol'    :fpol,  # Poloidal current function in m-T, $F=RB_T$ on flux grid
    'pres'    :pres,  # Plasma pressure in $nt/m^2$ on uniform flux grid
    'ffprim'  :ffprim,# $FF^\prime(\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid
    'pprime'  :pprime,# $P^\prime(\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid
    'psizr'   :psirz, # Poloidal flus in Webber/rad on the rectangular grid points
    'qpsi'    :qpsi,  # q values on uniform flux grid from axis to boundary
    'nbbbs'   :nbbbs, # Number of boundary points
    'limitr'  :limitr,# Number of limiter points
    'rbbbs'   :rbbbs, # R of boundary points in meter
    'zbbbs'   :zbbbs, # Z of boundary points in meter
    'rlim'    :rlim,  # R of surrounding limiter contour in meter
    'rlim'    :zlim,  # R of surrounding limiter contour in meter
    }
def geqdsk2gmsh(data,filename):
    f=open(filename,"w");
    
        
    return
#     plt.contour(psirz.reshape([nh,nw]),[(sibry-simag)/10.0*i+simag for i in range(10)])
#     
#     plt.plot((rlims-rleft)/rdim*nw,zlims/zdim*nh+nh/2)
#     
#     plt.plot((rbbbs-rleft)/rdim*nw,zbbbs/zdim*nh+nh/2)
#     
#     plt.show()

