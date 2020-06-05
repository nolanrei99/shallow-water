using LinearAlgebra
using SparseArrays
using Plots
gr()

const TINY = 1e-20

function vanleer(t)
    # flux-limiting function
    at = abs(t)
    return (t + at)/(1.0 + at)
end
#=
function vanleer(t)
    return 1.0
end
=#
function superbee(t)
    return max(0.0, min(1.0,2*t), min(t, 2.0))
end

"""
Performs a Givens rotation on the kth column h of an
upper Hessenberg matrix
c and s are the cosine and sine arrays of the previous Givens rotations
"""
function givens_h!(k, c, s, h)
    tmp = 0.0
    for i = 1:k-1
        tmp = h[i]
        h[i] = conj(c[i])*h[i] + conj(s[i])*h[i+1]
        h[i+1] = -s[i]*tmp + c[i]*h[i+1]
    end
    tmp = norm(h[k:k+1],2)
    if tmp != 0.0
        c[k] = h[k]/tmp
        s[k] = h[k+1]/tmp
    else
        c[k] = 1.0
        s[k] = 0.0
    end

    tmp = h[k]
    h[k] = conj(c[k])*h[k] + conj(s[k])*h[k+1]
    h[k+1] = -s[k]*tmp + c[k]*h[k+1]
    return nothing
end

"""
Arnoldi iteration
Calculates the (n+1)th Arnoldi vector of the mxm matrix A
Stores it in Q[:,n+1]
Stores the coefficients of the iteration in the (n+1)x1 vector h
"""
function arnoldi!(n, A, Q, h)
    # Arnoldi iteration
   mul!(view(Q,:,n+1),A,Q[:,n])
    for j = 1:n
        # MGSC
        h[j] = Q[:,j]'*Q[:,n+1]
        Q[:,n+1] .-= h[j].*Q[:,j]
    end
    h[n+1] = norm(Q[:,n+1])
    if n < size(A,2)
        Q[:,n+1] ./= h[n+1]
    else
        h[n+1] = 0
    end
    return nothing
end

"""
GMRES (Generalized Minimum RESidual) iterative method
Solves Ax = b iteratively to tolerance tol, using at most
nmax iterations
Overwrites the initial guess x with the approximate solution
"""
function gmres!(A::AbstractArray{T,2}, b::AbstractArray{T,1}, x::AbstractArray{T,1}, tol=1e-6, nmax=length(b)) where {T<:Number}
    Q = Array{T}(undef,length(b),nmax+1)
    R = zeros(T,nmax,nmax)		# IMPORTANT - getrs accesses the lower tri portion of R as well, so must be 0!
    h = Array{T}(undef,nmax+1)
    y = zeros(T,nmax+1)
    s = Array{T}(undef,nmax)
    c = Array{T}(undef,nmax)
	relres = Array{Float64}(undef,nmax)
    ipiv = vec(Array(1:nmax))
    tmp = 0.0
    nb = norm(b)
    r0 = b .- A*x
    y[1] = norm(r0)
    if y[1] != 0.0
        Q[:,1] .= r0./y[1]
    else
        return 1, 0.0
    end

    n = 1
    while true
        arnoldi!(n,A,Q,h)
        givens_h!(n,c,s,h)
        R[1:n,n] .= h[1:n]
        # Apply Givens rotation from triangularizing H to y
        tmp = y[n]
        y[n] = conj(c[n])*y[n] + conj(s[n])*y[n+1]
        y[n+1] = -s[n]*tmp + c[n]*y[n+1]
        # As we are minimizing the norm of the residual in Rx = y,
        # but R's last row is 0, the norm of the residual is the last element of y
        relres[n] = abs(y[n+1])/nb
        if relres[n] < tol || n == nmax
            break
        end
        n += 1
    end
    # Solving the triangular system to find the component of x in the Krylov subspace
    #ldiv!(R[1:n,1:n],view(y,1:n))
    LAPACK.getrs!('N',R[1:n,1:n],ipiv,view(y,1:n))
    x .+= Q[:,1:n]*y[1:n]
    return n, relres[1:n]
end


function lhs_mat!(nu, rhs, bdy, dx, dy, M, N)
    # Constructs the matrix representing the diffusion operator
    # and the rest of the terms of a Crank-Nicolson discretization
    # of the 2D kinematic equations for the ocean (KEO below)
    # Modifies rhs in place to reflect the impact of the Neumann 
    #   top-surface boundary conditions stored in bdy
    idx = (x,y) -> (x-1)*M + y
    dx_2 = 1.0/dx/dx
    dy_2 = 1.0/dy/dy
                     # diffusion   sum D_x u   identities for h and w
    ii = zeros(Int64,5*M*N - N   + 2*M*N     + 2*N)
    jj = zeros(Int64,5*M*N - N   + 2*M*N     + 2*N)
    vv = zeros(      5*M*N - N   + 2*M*N     + 2*N)
    ctr = 1

    # klooge - diffusion is coded with the wrong sign, so here switching sign of diffusion coeff.
    nu = -nu
    ## DIFFUSION MATRIX
    for y = 1:M
        for x = 1:N
            # self
            i = idx(x,y)
            ii[ctr] = i
            jj[ctr] = i
            vv[ctr] = 2.0*dx_2*nu + 2.0*dy_2*nu
            ctr += 1
            # left
            # Periodic boundary conditions
            ii[ctr] = i
            jj[ctr] = idx(mod(x-2,N)+1,y)
            vv[ctr] = -dx_2*nu
            ctr += 1
            # right
            # Periodic boundary conditions
            ii[ctr] = i
            jj[ctr] = idx(mod(x,N)+1,y)
            vv[ctr] = -dx_2*nu
            ctr += 1
            # top
            if y == 1
                # Tweak the right-hand side as well - Neumann
                # du/dz|z=0 is prescribed, so use it to calculate ghost node
                rhs[i] += dy_2*nu*(-dy*bdy[x])
                ii[ctr] = i
                jj[ctr] = idx(x,y)
                vv[ctr] = -dy_2*nu
                ctr += 1
            else
                ii[ctr] = i
                jj[ctr] = idx(x,y-1)
                vv[ctr] = -dy_2*nu
                ctr += 1
            end
            # bottom
            if y == M
                # Tweak the right-hand side instead - no-flux
                rhs[i] += dy_2*nu*0.0       # Arguable. This is implementing no-slip conditions in
                                            # the diffusion and no-flux in the advection. We'll see.
            else
                ii[ctr] = i
                jj[ctr] = idx(x,y+1)
                vv[ctr] = -dy_2*nu
                ctr += 1
            end
        end
    end
#=
    #### COUPLING TERMS
    # du/dt + ... = -g*dh/dx
    g = 9.81    #m/s^2
    for x = 1:N
        for y = 1:M
            i = idx(x,y)
            ii[ctr] = i
            jj[ctr] = M*N + x
            vv[ctr] = -g/dx
            ctr += 1
            ii[ctr] = i
            jj[ctr] = M*N + mod(x-2,N)+1
            vv[ctr] = g/dx
            ctr += 1
        end
    end
    
    # dh/dt = w
    for x = 1:N
        ii[ctr] = M*N + x
        jj[ctr] = M*N + N + x
        vv[ctr] = 1.0
        ctr += 1
    end
=#
    return ii,jj,vv,ctr
end

function convert_lhs!(ii, jj, vv, ctr0, dx, dy, dt, M, N)
    # Takes the output of lhs_mat! and adds the condition that w = ∫ -(∂_x u) dz
    # Also rescales by -dt/2 and adds identity matrix to yield the Crank-Nicolson dynamics matrix
    idx = (x,y) -> (x-1)*M + y
    ctr = copy(ctr0)      # preserving the initial value of ctr so that it can still be used

    # Rescaling matrix
    vv .*= (-0.5*dt)
    
    # Adding identity for w, h
    for i = 1:2N
        ii[ctr] = M*N+i
        jj[ctr] = M*N+i
        vv[ctr] = 1.0
        ctr += 1
    end

    # Adding integral
    for x = 1:N
        for y = 1:M
            ii[ctr] = M*N + N + x
            jj[ctr] = idx(x,y)
            vv[ctr] = -dy/dx
            ctr += 1
            ii[ctr] = M*N + N + x
            jj[ctr] = idx(mod(x,N)+1,y)
            vv[ctr] = dy/dx
            ctr += 1
        end
    end

    #println(ii,"\n")
    #println(jj,"\n")
    A = sparse(ii,jj,vv,(M+2)*N,(M+2)*N)
    # Adding identity to u section of matrix is easier in matrix form
    for i = 1:M*N
        A[i,i] += 1.0
    end
    return A
end

function convert_rhs!(ii, jj, vv, ctr0, dx, dy, dt, M, N)
    # Takes the output of lhs_mat! and tweaks it to serve as the RHS matrix in CN discretization
    # Also rescales by dt/2 and adds identity matrix to yield the Crank-Nicolson dynamics matrix
    idx = (x,y) -> (x-1)*M + y
    ctr = ctr0 + 0      # preserving the initial value of ctr so that it can still be used

    # Rescaling matrix
    vv .*= (0.5*dt)
    
    # Adding identity for h (not for w)
    for i = 1:N
        ii[ctr] = M*N+i
        jj[ctr] = M*N+i
        vv[ctr] = 1.0
        ctr += 1
    end

    # shorten ii, jj, and vv; the space for ∫ -(∂_x u) is not needed
    ii = ii[1:ctr-1]
    jj = jj[1:ctr-1]
    vv = vv[1:ctr-1]

    #println(ii,"\n")
    #println(jj)
    B = sparse(ii,jj,vv,(M+2)*N,(M+2)*N)

    # Adding identity to u section of matrix is easier in matrix form
    for i = 1:M*N
        B[i,i] += 1.0
    end
    return B
end

function keo(u0, h0, nu, dx, dz, dt, t0, T)
    # Solves (very simplified) Kinematic Equations of the Ocean in a 2D vertical slice
    # u0 is initial 2D velocity field
    # h0 is initial free surface deviation
    # nu is a viscosity parameter
    # (t0, T) is the interval over which equations should be integrated
    M,N = size(u0)

    u = 1.0*u0
    h = 1.0*h0
    t = 0.0+t0
    q = zeros((M+2)*N)      # solution vector
    q[1:(M*N)] .= reshape(u, (M*N,))
    u = reshape(view(q, 1:M*N), (M, N))
    q[M*N+1:M*N+N] .= h
    h = view(q, M*N+1:M*N+N)
    w = view(q, M*N+N+1:M*N+2N)

    bdy = zeros(N)      # top derivative is set to bdy; zero as placeholder
    rhs = zeros((M+2)*N)# right-hand side vector
    rhs_holder = zeros((M+2)*N)     # Holds effect of boundary conditions on rhs
    ii,jj,vv,ctr = lhs_mat!(nu, rhs_holder, bdy, dx, dz, M, N)
    rhs_holder .*= dt               # make it fit with rest of matrices A,B
    ii_ = copy(ii)
    jj_ = copy(jj)
    vv_ = copy(vv)
    A = convert_lhs!(ii,jj,vv,ctr,dx,dz,dt,M,N)
    B = convert_rhs!(ii_,jj_,vv_,ctr,dx,dz,dt,M,N)
    
    uFx = zeros(M,N)
    uFz = zeros(M+1,N)

    # split-explicit variables
    q1 = zeros(N)       # eigenvector of solution; (sqrt(gH), 1) in (U, h) with wavespeed sqrt(gH)
    q2 = zeros(N)       # eigenvector of solution; (-sqrt(gH), 1) with wavespeed -sqrt(gH)
    q1F = zeros(N)
    q2F = zeros(N)
    U = zeros(N)
    dt_ = 0.001*dt
    t_ = copy(t)
    d_ = dt_/dt
    q1accum = zeros(N)
    q2accum = zeros(N)
    g = 9.81    #m/s^2
    v = sqrt(g*dz*M)
    

    s = 0.0
    jj = 0
    jjj = 0

    # plotting code
    xx = dx*(0.5:1.0:N-0.5)
    idx = 0
    num = 10
    plt = 100

    while t < T
        # fast mode - evolution of h & w
        # Assumes u is only affected by eta in small timesteps, so can solve
        # for h's effect on w directly -- ∂w/∂t = gH(∂/∂x)^2 h
        
        # Calculate U on cell boundaries
        for j = 1:N
            U[j] = 0.0
            for i = 1:M
                U[j] += dz*u[i,j]
            end
        end

        # Interpolate U onto cell middles
        unew = 0.375*U[1] + 0.75*U[2] - 0.125*U[3]
        up = 0.0
        for j = 2:N
            up = unew
            unew = 0.375*U[j] + 0.75*U[mod(j,N)+1] - 0.125*U[mod(j+1,N)+1]
            U[j] = up
        end

        # Calculate q1 and q2 from U and h
        for j = 1:N
            q1[j] =  0.5/v*U[j] + 0.5*h[j]
            q2[j] = -0.5/v*U[j] + 0.5*h[j]
        end
        
        vv = 0.0
        t_ = 0.0
        while t_ < 0.5*dt
            for j = 1:N
                jj = mod(j-2,N)+1
                jjj = mod(j,N)+1
                vv = U[j] + v
                if vv > 0
                    theta = (q1[j]-q1[jj]+TINY)/(q1[jj]-q1[mod(j-3,N)+1]+TINY)
                    q1F[j] = vv*dt_*(q1[jj] - vanleer(theta)*0.5*(1.0-dt_*vv/dx)*(q1[jjj]-q1[j]))/dx
                else
                    theta = (q1[jjj]-q1[j]+TINY)/(q1[j]-q1[jj]+TINY)
                    q1F[j] = -vv*dt_*(q1[j] + vanleer(theta)*0.5*(1.0+dt_*vv/dx)*(q1[j]-q1[jj]))/dx
                end
                vv = U[j] - v
                if vv > 0
                    theta = (q2[j]-q2[jj]+TINY)/(q2[jj]-q2[mod(j-3,N)+1]+TINY)
                    q2F[j] = vv*dt_*(q2[jj] - vanleer(theta)*0.5*(1.0-dt_*vv/dx)*(q2[jjj]-q2[j]))/dx
                else
                    theta = (q2[jjj]-q2[j]+TINY)/(q2[j]-q2[jj]+TINY)
                    q1F[j] = vv*dt_*(q2[j] + vanleer(theta)*0.5*(1.0+dt_*vv/dx)*(q2[j]-q2[jj]))/dx
                end
            end
            for j = 1:N
                q1[j] += (q1F[j]-q1F[mod(j,N)+1])
                q2[j] += (q2F[j]-q2F[mod(j,N)+1])
            end
            t_ += dt_
        end
        q1accum .= 0.0
        q2accum .= 0.0
        while t_ < 1.5*dt
            for j = 1:N
                jj = mod(j-2,N)+1
                jjj = mod(j,N)+1
                theta = (q1[j]-q1[jj]+TINY)/(q1[jj]-q1[mod(j-3,N)+1]+TINY)
                q1F[j] = v*dt_*(q1[jj] - vanleer(theta)*0.5*(1.0-dt*v/dx)*(q1[jjj]-q1[j]))/dx
                theta = (q2[j]-q2[jj]+TINY)/(q2[jjj]-q2[j]+TINY)
                q2F[j] = -v*dt_*(q2[j] + vanleer(theta)*0.5*(1.0-dt*v/dx)*(q2[j]-q2[jj]))/dx
            end
            for j = 1:N
                q1[j] += (q1F[j]-q1F[mod(j,N)+1])
                q2[j] += (q2F[j]-q2F[mod(j,N)+1])
                q1accum[j] += d_*q1[j]
                q2accum[j] += d_*q2[j]
            end
            t_ += dt_
        end

        # Recombine q1 and q2 into U and h, and then calculate w from U
        for j = 1:N
            U[j] = v*(q1accum[j]-q2accum[j])
            h[j] = q1accum[j]+q2accum[j]
        end

        for j = 1:N
            w[j] = (U[j] - U[mod(j-2,N)+1])/dx
        end

        # slow mode - evolution of u
        for j = 1:N
            jj = mod(j-2,N)+1        # Periodic boundary conditions - j-1
            jjj = mod(j,N)+1           # j+1
            for i = 1:M
                s = 0.5*(u[i,j]^2 - u[i,jj]^2)/(u[i,j] - u[i,jj]+TINY) # Rankine-Hugoniot shock speed
                if s > 0
                    # upwind is to the left
                    s = abs(s)
                    theta = (u[i,jj]-u[i,mod((j-3),N)+1]+TINY)/(u[i,j]-u[i,jj]+TINY)
                    uFx[i,j] = (0.5*u[i,jj]^2 + g*h[jj] - vanleer(theta)*0.5*s*(1.0-dt*s/dx)*(u[i,jjj] - u[i,j]))/dx
                else
                    # upwind is to the right
                    s = abs(s)
                    theta = (u[i,jjj]-u[i,j]+TINY)/(u[i,j]-u[i,jj]+TINY)
                    uFx[i,j] = (0.5*u[i,j]^2 + g*h[j] - vanleer(theta)*0.5*s*(1.0-dt*s/dx)*(u[i,j] - u[i,jj]))/dx
                end
            end
        end

        for j = 1:N
            jjj = mod(j,N)+1           # j+1
            if w[j] < 0.0 
                theta = (u[2,j]-u[1,j]+TINY)/(u[3,j]-u[2,j]+TINY)
                uFz[2,j] = w[j]*u[2,j] - vanleer(theta)*0.5*w[j]*(1.0-dt*w[j]/dz)*(u[2,j]-u[1,j])
                for i = 3:M
                    theta = (u[i,j]-u[i-1,j]+TINY)/(u[i-1,j]-u[i-2,j]+TINY)
                    uFz[i,j] = w[j]*u[i,j] - vanleer(theta)*0.5*w[j]*(1.0-dt*w[j]/dz)*(u[i-1,j]-u[i-2,j])
                end
            else
                theta = (u[2,j]-u[1,j]+TINY)/(u[3,j]-u[2,j]+TINY)
                uFz[2,j] = w[j]*u[1,j] - vanleer(theta)*0.5*w[j]*(1.0-dt*w[j]/dz)*(u[2,j]-u[1,j])
                for i = 3:M
                    theta = (u[i,j]-u[i-1,j]+TINY)/(u[i-1,j]-u[i-2,j]+TINY)
                    uFz[i,j] = w[j]*u[i-1,j] - vanleer(theta)*0.5*w[j]*(1.0-dt*w[j]/dz)*(u[i,j]-u[i-1,j])
                end
            end
        end
#=
        if (idx%num == 0)
            display(plot(xx,q[M*N+1:M*N+N]))
        end
=#
        # Solve the system of equations
        mul!(rhs,B,q)
        
        for j = 1:N
            jjj = mod(j,N)+1           # j+1
            for i = 1:M
                rhs[(j-1)*M+i] += dt*(uFx[i,j]-uFx[i,jjj] + (uFz[i,j]-uFz[i+1,j])/dz)
            end
        end
        
        rhs += rhs_holder
        gmres!(A,rhs,q,1e-7,200)
        
        t += dt
        idx += 1

        if (idx%num == 0)
            display(plot(xx,hcat(q[1:M:M*N],q[M*N+1:M*N+N]),label=["surface velocity (m/s)" "surface deviation (m)"],xlabel="Horizontal position (m)"))
            if (idx%plt == 0)
                #savefig("M$(M)N$(N)dx$(dx)dz$(dz)dt$(dt)_$(idx).png")
            end
        end

    end
    return q
end

M = 50
N = 300
dx = 1e5/N
dz = 1e3/M
xx = dx*(0.5:1.0:N-0.5)
zz = dz*(0.5:1.0:M-0.5)
u0 = zeros(M,N)
u0[1:div(M,2),1:N] .= 1.0
h0 = zeros(N)
h0[1:div(N,2)] .= 1.0
nu = 1e-4
dt = 2.0e-1
t0 = 0.0
T = 50000*dt
q = keo(u0, h0, nu, dx, dz, dt, t0, T)
0

#plot(xx,q[M*N+1:M*N+N])
#plot(xx,q[1:M:M*N])
