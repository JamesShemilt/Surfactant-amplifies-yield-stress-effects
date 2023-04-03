%LateTimeAsympSolve
%March 2023

%Script to solve for the late-time asymptotic solution, up to O(1/t), as
%defined in section 3.2 of Shemilt et al. (2023) Surfactant amplifies
%yield-stress effects in the capillary instability of a film coating a
%tube. This code is used to produce data shown in figure 4. 

%Parameter values here are the ones used to produce figure 4 in Shemilt et
%al. (2023). This script calls the function Surf_thinfilm_evolution in
%order to run a simulation solving the full thin-film equations numerically
%to a long time. This numerical solution is used to produce guess functions
%for the BVP solver for the leading order and then O(1/t) late-time
%problems. 

B = 0.05;           %Capillary Bingham number
k = sqrt(2)/2;      %wavenumber
L = pi/k;           %Domain length
A = 0.25;           %Initial perturbation for simulation
N = 100;            %Number of grid points for simulation (can use smaller
                        %number to save time)
dz = pi/(k*(N-1));  %grid spacing
Z = 0:dz:L;         %Spatial grid for simulation
Ma = 0.5;           %Marangoni number
Tend = 1e4;         %End integration time for simulation
Ymin = 1e-8;         %Regularisation parameter for simulation
Tol = 1e-10;        %Tolerance for simulation
TolB = 1e-6;        %Tolerance for BVP solver
t = [0 Tend];       %Integration limits for simulation

%Get long time solution from simulation
Hsol = Surf_thinfilm_evolution(B,Ma,A,Ymin,N,Tol,t);


%%% Derivative matrices.
Dz = zeros(N);
Dzz = zeros(N);
Dzzz = zeros(N);

for n = 2:N-1
    Dz(n,n-1) = -1/(2*dz);
    Dz(n,n+1) = 1/(2*dz);
end

for n = 3:N-2
    Dzzz(n,n-2:n+2)= [-0.5, 1, 0,-1,0.5]/(dz^3);
end

Dzz(1,1:2) = [-2, 2]./(dz.^2);
for n = 2:N-1
    Dzz(n,n-1:n+1) = [1, -2, 1]./(dz.^2);
end
Dzz(N,N-1:N) = [2, -2]./(dz.^2);


%--% Using central finite differences and enforcing symmetry conditions to
%--% evaluate the second and penultimate poitns.
Dzzz(2,1:4) = [1, -0.5, -1, 0.5]/(dz^3);
Dzzz(N-1,N-3:N) = [-0.5, 1, 0.5,-1]/(dz^3);
Dzzz = sparse(Dzzz);
Dzz = sparse(Dzz);
Dz = sparse(Dz);

%Produce guess functions for the leading order problem
H = sol.y(1:N,end);
G = Ma*sol.y(N+1:end,end);
Hz = Dz*H;
Hzz = Dzz*H;
Hzzz = Dzzz*H;
u = cumtrapz(Z,H);%integral of H
Kz = -Hz - Hzzz;

%Construct BVProblem for leading order problem
xmesh = linspace(0,L,150);
solinit0 = bvpinit(xmesh, @(z) guess(z,H,Hz,Hzz,u,Z));
options = bvpset('RelTol',TolB,'AbsTol',TolB);
sol0 = bvp4c(@(z,y) bvpfn0(z,y,2*B), @(ya,yb)bcfn0(ya,yb,L), solinit0,options);

%Define guess functions for the first order problem
h0 = deval(sol0,Z);
modKz = abs(Kz);
Gz = Dz*G;
Y = 0*H;
Y(2:N-1) = H(2:N-1) - B./modKz(2:N-1)+Gz(2:N-1)./Kz(2:N-1);
Y = Y.*Tend;
Y2 = Y.^2;
G1 = (G - Ma + L*B/2 - B.*Z').*Tend;
G1z = (Gz - B).*Tend;
G1z(1) = 0;
G1z(end) = 0;
H1 = (H' - h0(1,:)).*Tend;
H1z = (Hz' - h0(2,:)).*Tend;
H1zz = (Hzz' - h0(3,:)).*Tend;
W1 = B.*Y2./H - H.*G1z.*G1z./(2*B);

%Construct BVProblem for first order problem
solinit1 = bvpinit(xmesh,@(z) guess1(z,H1,H1z,H1zz,Y2,G1,W1,Z));
sol1 = bvp4c(@(x,y)bvpfn1(x,y,sol0,B,Ma-B*L/2),@bcfn1,solinit1,options);

function dydx = bvpfn0(z,y,b)
%Function defining the system of ODEs for the leading order problem
%Solves for the vector y = [H0 H0z H0zz u], where u is the integral of H
dydx = [y(2) 
        y(3) 
        (b/y(1))-y(2)
        y(1)
        ];
end

function res = bcfn0(ya,yb,L)
%Function enforcing the boundary conditions in the leading order problem
res = [ya(2)
       yb(2)
       ya(4)
       yb(4)-L
       ];
end

function g = guess(z,H,Hz,Hzz,u,Z)
%Guess function for the leading order problem
    g = zeros(4,1);
    g(1) = spline(Z,H,z);
    g(2) = spline(Z,Hz,z);
    g(3) = spline(Z,Hzz,z);
    g(4) = spline(Z,u,z);
end

function dydx = bvpfn1(z,y,sol,B,MGam0)
%Function defining the system of ODEs for the first order problem
%Solves for the vector y = [H1 H1z H1zz Y2 Gamma1 W1] where Y2 is Y-^2. 
H0 = deval(sol,z);
dydx = [y(2)                                        %H1z
        y(3)                                        %H1zz
        -y(2) + (sqrt(4*B*B*y(4)/H0(1)^2 - 4*B*y(6)/H0(1))/H0(1)) + (sqrt(y(4))-y(1))*2*B/(H0(1)^2) %H1zzz
        y(1)/B                                      %Y2z
        sqrt(4*B*B*y(4)/H0(1)^2 - 4*B*y(6)/H0(1))   %Gamma1z (from definition of w_s)
        (y(5)-B*y(6))/(MGam0+B*z)                   %W1z (from surfactant transport equation)
        ];
end

function res = bcfn1(ya,yb)
%Function enforcing the boundary conditions for the first order problem
res = [ya(2)        %H1z=0 at z=0
       yb(2)        %H1z=0 at z=L
       ya(4)        %Ym1=0 at z=0
       yb(4)        %Ym1=0 at z=L
       ya(6)        %W1=0 at z=0
       yb(6)        %W1=0 at z=L
       ];
end

function g = guess1(z,H1,H1z,H1zz,Y2,MG1,W1,Z)
%Guess function for the solver for the first order problem
    g = zeros(6,1);
    g(1) = spline(Z,H1,z);
    g(2) = spline(Z,H1z,z);
    g(3) = spline(Z,H1zz,z);
    g(4) = spline(Z,Y2,z);
    g(5) = spline(Z,MG1,z);
    g(6) = spline(Z,W1,z);
end