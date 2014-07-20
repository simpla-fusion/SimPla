import matplotlib.animation as animation
import h5py as H5



f1=H5.File("data_dump/geqdsk0027.h5","r")

r=linspace(1.2,2.8,128)
q,r=meshgrid(q,r)
q=linspace(0,6.28/4,128)
q,r=meshgrid(q,r)

fig=figure()

def animate(iter):
    clf();
    contour(r*sin(q),r*cos(q),f1["/Save/E1"][iter+1,:,:,2])
    
anim = animation.FuncAnimation(fig,animate ,frames=range(720),interval=20,blit=False,repeat=True)
 
metadata = dict(title='Movie Test', artist='YuZhi',
        comment='LHW')
 
writer =animation.FFMpegWriter(fps=30, metadata=metadata)
 
anim.save("a.mp4",writer)
