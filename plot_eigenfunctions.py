import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

filenames = ['eigenfunction.h5','eigenfunctionNP.h5']
variable = ['UltraS','y']#,'UltraSl']

data={}
for filei in filenames:
    with h5py.File(filei,'r') as f:
        for var in variable:
            data[filei+'/'+var] = f[var][:,0] + 1.j*f[var][:,1]
e0 = data[filenames[0]+'/'+variable[0]]
e1 = data[filenames[1]+'/'+variable[0]]
y0 = data[filenames[0]+'/'+variable[1]].real
y1 = data[filenames[1]+'/'+variable[1]].real
#e0l = data[filenames[0]+'/'+variable[2]]
#e1l = data[filenames[1]+'/'+variable[2]]
ny0=y0.shape[0]
ny1=y1.shape[0]
norm0 = e0[np.argmax(np.abs(e0[:ny0]))]
norm1 = e1[np.argmax(np.abs(e1[:ny1]))]
#norm0l = e0l[np.argmax(np.abs(e0l[:ny0]))]
#norm1l = e1l[np.argmax(np.abs(e1l[:ny1]))]
u0 ,v0 ,w0 ,p0 ,au0 ,av0 ,aw0 ,ap0  = e0 .reshape(8,ny0)/norm0
#u0l,v0l,w0l,p0l,au0l,av0l,aw0l,ap0l = e0l.reshape(8,ny0)/norm0l
u1 ,v1 ,w1 ,p1 ,du1 ,dv1 ,dw1 ,dp1 ,au1 ,av1 ,aw1 ,ap1 ,dau1 ,dav1 ,daw1 ,dap1  = e1 .reshape(16,ny1)/norm1
#u1l,v1l,w1l,p1l,du1l,dv1l,dw1l,dp1l,au1l,av1l,aw1l,ap1l,dau1l,dav1l,daw1l,dap1l = e1l.reshape(16,ny1)/norm1l

fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=4,tight_layout=True,sharey=True)
ax[0,0].plot(np.abs(u0),y0,'-',label='LST')
ax[0,0].plot(u0.real,y0,'-',label='real(LST)')
ax[0,0].plot(u0.imag,y0,'-',label='imag(LST)')
ax[0,0].plot(np.abs(u1),y1,'--',label='LSTNP')
ax[0,0].plot(u1.real,y1,'--',label='real(LSTNP)')
ax[0,0].plot(u1.imag,y1,'--',label='imag(LSTNP)')
ax[0,1].plot(np.abs(v0),y0,'-',label='LST')
ax[0,1].plot(v0.real,y0,'-',label='real(LST)')
ax[0,1].plot(v0.imag,y0,'-',label='imag(LST)')
ax[0,1].plot(np.abs(v1),y1,'--',label='LSTNP')
ax[0,1].plot(v1.real,y1,'--',label='real(LSTNP)')
ax[0,1].plot(v1.imag,y1,'--',label='imag(LSTNP)')
ax[0,2].plot(np.abs(w0),y0,'-',label='LST')
ax[0,2].plot(w0.real,y0,'-',label='real(LST)')
ax[0,2].plot(w0.imag,y0,'-',label='imag(LST)')
ax[0,2].plot(np.abs(w1),y1,'--',label='LSTNP')
ax[0,2].plot(w1.real,y1,'--',label='real(LSTNP)')
ax[0,2].plot(w1.imag,y1,'--',label='imag(LSTNP)')
ax[0,3].plot(np.abs(p0),y0,'-',label='LST')
ax[0,3].plot(p0.real,y0,'-',label='real(LST)')
ax[0,3].plot(p0.imag,y0,'-',label='imag(LST)')
ax[0,3].plot(np.abs(p1),y1,'--',label='LSTNP')
ax[0,3].plot(p1.real,y1,'--',label='real(LSTNP)')
ax[0,3].plot(p1.imag,y1,'--',label='imag(LSTNP)')
ax[1,0].plot(np.abs(au0),y0,'-',label='LST')
ax[1,0].plot(au0.real,y0,'-',label='real(LST)')
ax[1,0].plot(au0.imag,y0,'-',label='imag(LST)')
ax[1,0].plot(np.abs(au1),y1,'--',label='LSTNP')
ax[1,0].plot(au1.real,y1,'--',label='real(LSTNP)')
ax[1,0].plot(au1.imag,y1,'--',label='imag(LSTNP)')
ax[1,1].plot(np.abs(av0),y0,'-',label='LST')
ax[1,1].plot(av0.real,y0,'-',label='real(LST)')
ax[1,1].plot(av0.imag,y0,'-',label='imag(LST)')
ax[1,1].plot(np.abs(av1),y1,'--',label='LSTNP')
ax[1,1].plot(av1.real,y1,'--',label='real(LSTNP)')
ax[1,1].plot(av1.imag,y1,'--',label='imag(LSTNP)')
ax[1,2].plot(np.abs(aw0),y0,'-',label='LST')
ax[1,2].plot(aw0.real,y0,'-',label='real(LST)')
ax[1,2].plot(aw0.imag,y0,'-',label='imag(LST)')
ax[1,2].plot(np.abs(aw1),y1,'--',label='LSTNP')
ax[1,2].plot(aw1.real,y1,'--',label='real(LSTNP)')
ax[1,2].plot(aw1.imag,y1,'--',label='imag(LSTNP)')
ax[1,3].plot(np.abs(ap0),y0,'-',label='LST')
ax[1,3].plot(ap0.real,y0,'-',label='real(LST)')
ax[1,3].plot(ap0.imag,y0,'-',label='imag(LST)')
ax[1,3].plot(np.abs(ap1),y1,'--',label='LSTNP')
ax[1,3].plot(ap1.real,y1,'--',label='real(LSTNP)')
ax[1,3].plot(ap1.imag,y1,'--',label='imag(LSTNP)')
ax[0,0].legend(loc='best',numpoints=1)
ax[0,0].set_ylabel(r'$y$')
ax[1,0].set_ylabel(r'$y$')
ax[0,0].set_xlabel(r'$\hat{u}$')
ax[0,1].set_xlabel(r'$\hat{v}$')
ax[0,2].set_xlabel(r'$\hat{w}$')
ax[0,3].set_xlabel(r'$\hat{p}$')
ax[1,0].set_xlabel(r'$\alpha\hat{u}$')
ax[1,1].set_xlabel(r'$\alpha\hat{v}$')
ax[1,2].set_xlabel(r'$\alpha\hat{w}$')
ax[1,3].set_xlabel(r'$\alpha\hat{p}$')
fig.show()

if False:
    fig,ax = plt.subplots(figsize=(8,8),nrows=2,ncols=4,tight_layout=True)
    ax[0,0].plot(np.abs(u0l),y0,'-',label='LST')
    ax[0,0].plot(u0l.real,y0,'-',label='real(LST)')
    ax[0,0].plot(u0l.imag,y0,'-',label='imag(LST)')
    ax[0,0].plot(np.abs(u1l),y1,'--',label='LSTNP')
    ax[0,0].plot(u1l.real,y1,'--',label='real(LSTNP)')
    ax[0,0].plot(u1l.imag,y1,'--',label='imag(LSTNP)')
    ax[0,1].plot(np.abs(v0l),y0,'-',label='LST')
    ax[0,1].plot(v0l.real,y0,'-',label='real(LST)')
    ax[0,1].plot(v0l.imag,y0,'-',label='imag(LST)')
    ax[0,1].plot(np.abs(v1l),y1,'--',label='LSTNP')
    ax[0,1].plot(v1l.real,y1,'--',label='real(LSTNP)')
    ax[0,1].plot(v1l.imag,y1,'--',label='imag(LSTNP)')
    ax[0,2].plot(np.abs(w0l),y0,'-',label='LST')
    ax[0,2].plot(w0l.real,y0,'-',label='real(LST)')
    ax[0,2].plot(w0l.imag,y0,'-',label='imag(LST)')
    ax[0,2].plot(np.abs(w1l),y1,'--',label='LSTNP')
    ax[0,2].plot(w1l.real,y1,'--',label='real(LSTNP)')
    ax[0,2].plot(w1l.imag,y1,'--',label='imag(LSTNP)')
    ax[0,3].plot(np.abs(p0l),y0,'-',label='LST')
    ax[0,3].plot(p0l.real,y0,'-',label='real(LST)')
    ax[0,3].plot(p0l.imag,y0,'-',label='imag(LST)')
    ax[0,3].plot(np.abs(p1l),y1,'--',label='LSTNP')
    ax[0,3].plot(p1l.real,y1,'--',label='real(LSTNP)')
    ax[0,3].plot(p1l.imag,y1,'--',label='imag(LSTNP)')
    ax[1,0].plot(np.abs(au0l),y0,'-',label='LST')
    ax[1,0].plot(au0l.real,y0,'-',label='real(LST)')
    ax[1,0].plot(au0l.imag,y0,'-',label='imag(LST)')
    ax[1,0].plot(np.abs(au1l),y1,'--',label='LSTNP')
    ax[1,0].plot(au1l.real,y1,'--',label='real(LSTNP)')
    ax[1,0].plot(au1l.imag,y1,'--',label='imag(LSTNP)')
    ax[1,1].plot(np.abs(av0l),y0,'-',label='LST')
    ax[1,1].plot(av0l.real,y0,'-',label='real(LST)')
    ax[1,1].plot(av0l.imag,y0,'-',label='imag(LST)')
    ax[1,1].plot(np.abs(av1l),y1,'--',label='LSTNP')
    ax[1,1].plot(av1l.real,y1,'--',label='real(LSTNP)')
    ax[1,1].plot(av1l.imag,y1,'--',label='imag(LSTNP)')
    ax[1,2].plot(np.abs(aw0l),y0,'-',label='LST')
    ax[1,2].plot(aw0l.real,y0,'-',label='real(LST)')
    ax[1,2].plot(aw0l.imag,y0,'-',label='imag(LST)')
    ax[1,2].plot(np.abs(aw1l),y1,'--',label='LSTNP')
    ax[1,2].plot(aw1l.real,y1,'--',label='real(LSTNP)')
    ax[1,2].plot(aw1l.imag,y1,'--',label='imag(LSTNP)')
    ax[1,3].plot(np.abs(ap0l),y0,'-',label='LST')
    ax[1,3].plot(ap0l.real,y0,'-',label='real(LST)')
    ax[1,3].plot(ap0l.imag,y0,'-',label='imag(LST)')
    ax[1,3].plot(np.abs(ap1l),y1,'--',label='LSTNP')
    ax[1,3].plot(ap1l.real,y1,'--',label='real(LSTNP)')
    ax[1,3].plot(ap1l.imag,y1,'--',label='imag(LSTNP)')
    ax[0,0].legend(loc='best',numpoints=1)
    ax[0,0].set_ylabel(r'$y$')
    ax[1,0].set_ylabel(r'$y$')
    ax[0,0].set_xlabel(r'$\hat{u}$')
    ax[0,1].set_xlabel(r'$\hat{v}$')
    ax[0,2].set_xlabel(r'$\hat{w}$')
    ax[0,3].set_xlabel(r'$\hat{p}$')
    ax[1,0].set_xlabel(r'$\alpha\hat{u}$')
    ax[1,1].set_xlabel(r'$\alpha\hat{v}$')
    ax[1,2].set_xlabel(r'$\alpha\hat{w}$')
    ax[1,3].set_xlabel(r'$\alpha\hat{p}$')
    fig.show()

# show inner products
#print(UltraS@(UltraS.conj()))
#print(Physical@(Physical.conj()))
#print(UltraSl@(UltraSl.conj()))
#print(Physicall@(Physicall.conj()))

#Re = 1000.0/1.7208
#i = 1.j
#O = np.zeros((ny,ny))
#O4 = np.zeros((4*ny,4*ny))
#I = np.eye(ny)
#I4 = np.eye(4*ny)
#L2 = np.block([
    #[-I, O, O, O],
    #[ O,-I, O, O],
    #[ O, O,-I, O],
    #[ O, O, O, O],
    #])
#M = np.block([
    #[I4, O4],
    #[O4,-L2]
    #])
#dLdomega4 = np.block([
    #[i*Re*I, O, O, O],
    #[O, i*Re*I, O, O],
    #[O, O, i*Re*I, O],
    #[O, O,      O, O],
    #])
#dLdomega = np.block([
    #[O4, O4],
    #[dLdomega4, O4]
    #])

#def inn(a,b):
    #return (a)@(b.conj())

#print('UltraS cg = ',inn(UltraS,M@UltraSl)/inn(dLdomega@UltraS,UltraSl))
#print('Physical cg = ',inn(Physical,M@Physicall)/inn(dLdomega@Physical,Physicall))
#print(inn(UltraS,UltraSl))
#print(inn(UltraS,M@UltraSl))
#print(inn(M@UltraS,UltraSl))
#print(inn(Physical,Physicall))
#print(inn(Physical,M@Physicall))
#print(inn(M@Physical,Physicall))
#tmp = inn(UltraS,M@UltraSl)
#UltraS2 = UltraS/tmp
#tmp = inn(Physical,M@Physicall)
#Physical2 = Physical/tmp
#print(inn(UltraS2,M@UltraSl))
#print(inn(Physical2,M@Physicall))
#print('UltraS2 cg = ',inn(UltraS2,M@UltraSl)/inn(dLdomega@UltraS2,UltraSl))
#print('UltraS2 lchange cg = ',inn(UltraS2,M@Physicall)/inn(dLdomega@UltraS2,Physicall))
#print('Physical2 cg = ',inn(Physical2,M@Physicall)/inn(dLdomega@Physical2,Physicall))

tmp0 = e0/norm0
tmp1 = e1/norm1
#tmp1l = UltraSl/norm1l
#tmp2l = Physicall/norm2l

#print('error = ',np.sum(np.abs(tmp0[:4*ny0]-tmp1[:4*ny1])))
#print('errorl = ',np.sum(np.abs(tmp1l-tmp2l)))


input('Enter to Exit')
