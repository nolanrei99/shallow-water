using LinearAlgebra
using SparseArrays
using Base.Threads
using PyPlot

function vanleer(t)
    # flux-limiting function
    at = abs(t)
    return (t + at)/(1.0 + at)
end

function superbee(t)
    return max(0.0, min(1.0,2*t), min(t, 2.0))
end

function advec2d(phi0, ufun, vfun, xx, yy, t0, T, dt)
    # Advection in a 2D box, assuming no diffusion and no flow through the sides of the box
    M,N = size(phi0)
    phi = 1.0*phi0
    Fx = zeros(M,N+1)
    Fy = zeros(M+1,N)
    t = t0+0.0
    tiny = 1e-20
    u = zeros(M,N+1)
    v = zeros(M+1,N)

    ct = 0
    num = 1
    xplt = 0.5*(xx[:,1:N]+xx[:,2:N+1])
    yplt = 0.5*(yy[1:M,:]+yy[2:M+1,:])
    #plot_surface(xplt,yplt,phi)
    pcolormesh(phi)
    show()

    while t < T
        u .= ufun(t)
        v .= vfun(t)
        for j = 1:N
            @simd for i = 1:M
                if j < N
                    # Flux in x
                    if j == 1
                        theta = 1.0
                    else
                        theta = (phi[i,j]-phi[i,j-1]+tiny)/(phi[i,j+1]-phi[i,j]+tiny)
                    end
                    dx = xx[i,j+1]-xx[i,j]
                    udt = u[i,j+1]*dt
                    if udt >= 0
                        # upwind is j
                        Fx[i,j+1] = udt*(phi[i,j] + 0.5*superbee(theta)*(1.0-udt/dx)*(phi[i,j+1]-phi[i,j]))
                    else
                        # upwind is j+1
                        if j > 1
                            Fx[i,j+1] = udt*(phi[i,j+1] + 0.5*superbee(theta)*(1.0-udt/dx)*(phi[i,j]-phi[i,j-1]))
                        else
                            Fx[i,j+1] = udt*(phi[i,j+1] + 0.5*superbee(theta)*(1.0-udt/dx)*(phi[i,j+1]-phi[i,j]))
                        end
                    end
                end

                if i < M
                    # Flux in y
                    if i == 1
                        theta = 1.0
                    else
                        theta = (phi[i,j]-phi[i-1,j]+tiny)/(phi[i+1,j]-phi[i,j]+tiny)
                    end
                    dy = yy[i+1,j]-yy[i,j]
                    udt = v[i+1,j]*dt
                    if udt >= 0
                        # upwind is i
                        Fy[i+1,j] = udt*(phi[i,j] + 0.5*superbee(theta)*(1.0-udt/dy)*(phi[i+1,j]-phi[i,j]))
                    else
                        # upwind is i+1
                        if i > 1
                            Fy[i+1,j] = udt*(phi[i+1,j] + 0.5*superbee(theta)*(1.0-udt/dy)*(phi[i,j]-phi[i-1,j]))
                        else
                            Fy[i+1,j] = udt*(phi[i+1,j] + 0.5*superbee(theta)*(1.0-udt/dy)*(phi[i+1,j]-phi[i,j]))
                        end
                    end
                end                
            end
        end

        for j = 1:N
            @simd for i = 1:M
                dx = xx[i,j+1]-xx[i,j]
                dy = yy[i+1,j]-yy[i,j]
                phi[i,j] += ((Fx[i,j] - Fx[i,j+1])/dy + (Fy[i,j] - Fy[i+1,j])/dx)
            end
        end
        
        t += dt
        ct += 1

        if (ct % num) == 0
            cla()
            #plot_surface(xplt,yplt,phi)
            pcolormesh(phi)
            show()
            pause(0.001)
        end
    end
    return phi
end


t0 = 0.0
T = 4.0
dt = 0.001


N = 256
dx = 1/N
dy = 1/N
xx = ones(N)*transpose(dx*(0:N))
yy = (dy*(0:N))*transpose(ones(N))
xplt = ones(N+1)*transpose(dx*(0.5:1.0:N-0.5))
yplt = (dy*(0.5:1.0:N-0.5))*transpose(ones(N+1))
u = -(sin.(pi*xx).^2).*sin.(2*pi*yplt)
v = sin.(2*pi*xplt).*sin.(pi*yy).^2
function ufun(t)
    #return cos(2*pi*t)*u
    return u
end
function vfun(t)
    #return cos(2*pi*t)*v
    return v
end
phi0 = zeros(N,N)
phi0[50:100,50:100] .= 1.0
@time phi = advec2d(phi0,ufun,vfun,xx,yy,t0,T,dt)

0.0
