-- Example Config file .------------------- --------
		
UNIT_DIMENSIONS="NATURE"   
        
DESCRIPTION="bala bala bong" -- description or other text things.   
        
DIMS={17,107,107}  -- number of grid, now only first dimension is valid      
        
GW= {3,3,3}  -- width of ghost points           
        
LENGTH={2*math.pi/(DIMS[1]-GW[1]*2-1)*(DIMS[1]-1),   
    4*math.pi/(DIMS[2]-GW[2]*2-1)*(DIMS[2]-1),   
    4*math.pi/(DIMS[3]-GW[3]*2-1)*(DIMS[3]-1)}  
        
-- length of simulation domain    
            
DT=0.5*LENGTH[3]/(DIMS[3]-1)    -- time step     
        
LOAD_FIELDS={E1=1,B0=2,N0=0}          
        
SP_LIST={"ele"}  
-- the list of species in the simulation       
            
SPECIES= -- predefine species        
{               
 ele={desc="ele" ,Z=-1.0, m=1.0, ns=1.0,  engine="ColdFluid",pic=0 ,T=0},  
        
 H_c={desc="H"    ,Z=1.0, m=1,    ns=1.0,     engine="ColdFulid",pic=0,T=0},      
            
 H_g={desc="H"    ,Z=1.0,   m=1,    ns=1.0, engine="GyroGauge", T=1.0e-4, numOfMate=20,    PIC=10}, 
} 
            
B0={}  -- initial/background magnetic field  
E1={}  -- initial/background electric field is zero if E0==null       
N0={}  -- initial/background density field of  electron;        


rx=(DIMS[2]-1)/2.0
ry=(DIMS[3]-1)/2.0
 
for k=0,DIMS[1] -1 do       
 for j=0,DIMS[2] -1 do    
  for i=0,DIMS[3] -1 do   
     s=k*DIMS[2]*DIMS[3]+ j*DIMS[3]+i    
     N0[s] = 6.0
     B0[s*3+0] = 0.1*(j -ry)/ry*2.0
     B0[s*3+1]= -0.1*(i-rx)/rx*2.0    
     B0[s*3+2]= 1.0-0.2*(i-rx)/rx*2.0     
        
  end   
 end   
end    
  
--[[  uncomment this line, if you need Cycle BC.
-- set BC(boundary condition), now only first two  are valid        
-- BC >= GW              
BC= {            
        -1,-1, -- direction x 
        -1,-1, -- direction y 
        -1,-1 -- direction z 
        }           
for x=0,DIMS[1] -1 do        
 for y=0,DIMS[2] -1 do   
  for z=0,DIMS[3] -1 do    
    s=x*DIMS[2]*DIMS[3]+ y*DIMS[3]+z   
    qx = (x-GW[1])%(DIMS[1]-2*GW[1]-1)/(DIMS[1]-2*GW[1]-1) *2.0 *math.pi   
    qy = (y-GW[2])%(DIMS[2]-2*GW[2]-1)/(DIMS[2]-2*GW[2]-1) *2.0 *math.pi   
    qz = (z-GW[3])%(DIMS[3]-2*GW[3]-1)/(DIMS[3]-2*GW[3]-1) *2.0 *math.pi   
    a=0.0   
    for kx=1,1 do   
     for ky=1,1 do   
        a=a+math.sin(qx*kx)*math.sin(qy*ky)   
     end   
    end   
   
    E0[s*3+0]=0.0    
    E0[s*3+1]=1.0e-8 *a   
    E0[s*3+2]=1.0e-8 *a   
  end   
 end   
end   
--]]          
        
---[[ uncomment this line, if you need absorbing condition.
BC= {           
     -1,-1, -- direction z
     1,1, -- direction y
     1,1 -- direction x
     }           
Srcf0=1.9              
Srckx=0           
Srctau=0.5              
        
function JSrc(t)             
     alpha=t*Srcf0 - Srctau*math.pi*2.0       
     res={}              
     if alpha<0 then             
           a=math.sin(alpha)*math.exp(-alpha*alpha)      
     else              
       a=math.sin(alpha)         
     end   
     for k=0, DIMS[1]-1 do
     	s=k*DIMS[2]*DIMS[3]+ (DIMS[2]-1)/2*DIMS[3]+10   
     	res[s*3+1]= 1.0e-8*math.sin(2.0*math.pi*(k-GW[1])/(DIMS[1]-GW[1]*2-1))*a   
     end          
     return res          
end               
--]]          
-- The End --------------------------------------- 
