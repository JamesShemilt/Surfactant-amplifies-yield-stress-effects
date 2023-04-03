%Surf_longwave_evolution
%March 2023

%Function to used to solve long-wave evolution equations in Shemilt et al. 
%(2023) Surfactant amplifies yield-stress effects in the capillary 
%instability of a film coating a tube. This code is used to get data shown
%in figures 7, 8(a) and 9. 

%The input parameters are: B, capillary Bingham number; Ma, Marangoni
%number; A, initial perturbation amplitude; Eps, thickness ration (epsilon)
%Ymin, regularisation parameter; N, number of finite difference grid 
%points; Tol, error tolerance in ODE solver; Hmax, value of H at which time
%the simulation is stopped Tend, end time for integration. Note that the 
%capillary Bingham and Marangoni numbers used are the unscaled version 
%defined in equations (2.18) and (2.20) of Shemilt et al. (2023). 

function solR = Surf_longwave_evolution(B,Ma,A,Eps,Ymin,N,Tol,Hmax,Tend)

k = sqrt(2)/2;              %wavenumber
L = pi/k;                   %domain length
Bh = B;                     %capillary Bingham number
Mah = Ma;                   %Marangoni number
Tend = Tend/Eps^3;          %rescaled end-time for integration
dz = L/(N-1);               %grid spacing
z = 0:dz:L;                 %finite difference grid

%Initial conditions for R and Gamma
R0 = sqrt((1-Eps)^2-(Eps^2)*(A^2)/2) + Eps.*A.*cos(pi.*z./L);
G0 = R0.*0 + 1;

%--% Dz, Dz and Dzzz are 1st, 2nd and 3rd derivative matrices which enforce
%symmetry boundary conditions.
Dz = zeros(N);
Dzz = zeros(N);
Dzzz = zeros(N);
for n = 2:N-1
    Dz(n,n-1) = -1/(2*dz);
    Dz(n,n+1) = 1/(2*dz);
end

for n = 2:N-1
    Dzz(n,n-1) = 1/(dz^2);
    Dzz(n,n) = -2/(dz^2);
    Dzz(n,n+1) = 1/(dz^2);
end
Dzz(1,1) = -2/(dz^2);
Dzz(1,2) = 2/(dz^2);
Dzz(N,N) = -2/(dz^2);
Dzz(N,N-1) = 2/(dz^2);

for n = 3:N-2
    Dzzz(n,n-2:n+2)= [-0.5, 1, 0,-1,0.5]/(dz^3);
end
Dzzz(2,1:4) = [1, -0.5, -1, 0.5]/(dz^3);
Dzzz(N-1,N-3:N) = [-0.5, 1, 0.5,-1]/(dz^3);

%Dzq is a 1st derivative matrix which enforces the flux boundary condition.
Dzq = zeros(N);
Dzq(1,2) = 1/dz;
for n = 2:N-1
    Dzq(n,n-1) = -1/(2*dz);
    Dzq(n,n+1) = 1/(2*dz);
end
Dzq(N,N-1) = -1/dz;

Dzzz = sparse(Dzzz);
Dzz = sparse(Dzz);
Dz = sparse(Dz);

%%% Sparsity matrix provides a Jacobian pattern, with '1' where there might
%%% be a non-zero entry in the Jacobian. 
N2 = 2*N;
Sparsity = zeros(N2);
Sparsity(1:4,1:6) = 1;
Sparsity(1:4,N+1:N+6) = 1;
Sparsity(N+1:N+4,1:6) = 1;
Sparsity(N-4:N+4,N-6:N+6) = 1;
Sparsity(2*N-4:2*N,2*N-6:2*N) = 1;
Sparsity(N-4:N,2*N-6:2*N) = 1;
Sparsity(2*N-4:2*N,N-6:N) = 1;
for n = 4:N-4
    Sparsity(n,n-3:n+3) = 1;
    Sparsity(n,n+N-2:n+N+2) = 1;
    Sparsity(n+N,n-3:n+3) = 1;
    Sparsity(n+N,n+N-2:n+N+2) = 1;
end

if B == 0 && Ma == 0
    Sparsity = zeros(N2);
Sparsity(1:4,1:6) = 1;
Sparsity(N-4:N,N-6:N) = 1;
for n = 4:N-4
    Sparsity(n,n-3:n+3) = 1;
end
end

Sparsity = sparse(Sparsity);

%Time span of integration
tspan = [0 Tend];

%Options for ode15s
options = odeset('RelTol',Tol,'AbsTol',Tol, 'Vectorized', 'on', 'JPattern', Sparsity,'Events',@EventFn);

%Solve ODE system derived from finite differencing the evolution equation
ICvector = [R0.^2 (R0.*G0)];%initial conditions vector for R^2 and R*Gamma
solR = ode15s(@deriv,tspan,ICvector,options);

function dR2 = deriv(~,R2)
%Function defining the ODE system from finite differencing the long-wave 
%evolution equation
        
        %Define R, R*Gamma and Gamma_z from the solution
        R = sqrt(R2(1:N,:));
        RG = R2(N+1:end,:);
        Gz = Dz*(RG./R);
        
        %Define the pressure gradient
        Rz = Dz*R;
        S2 = Rz.*Rz + 1;
        K = ((1./R) - (Dzz*R)./(S2))./sqrt(S2);
        Kz = -Rz./((R.*R).*sqrt(S2))-Rz.*(Dzz*R)./(R.*(S2.^1.5));
        Kz = Kz - (Dzzz*R)./(S2.^1.5) + 3.*Rz.*(Dzz*R).*(Dzz*R)./(S2.^2.5);
        pz = -Kz.*(1 + Mah.*(1-RG./R)) + K.*Mah.*Gz;
        
        modpz = abs(pz);%absolute value of pz
        sgnpz = sign(pz);%sign of pz
        
        %Define Psiplus and Psiminus
        Psip = 0*R +1;                      %Initialisation
        Psim = 0*R;                         %Initialisation
        Psip(2:N-1,:) = Bh./modpz(2:N-1,:) + real(sqrt(R(2:N-1,:).^2 + (Bh./pz(2:N-1,:)).^2 - 2.*R(2:N-1,:).*Mah.*Gz(2:N-1,:)./pz(2:N-1,:)));
        Psim(2:N-1,:) = -Bh./modpz(2:N-1,:) + real(sqrt(R(2:N-1,:).^2 + (Bh./pz(2:N-1,:)).^2 - 2.*R(2:N-1,:).*Mah.*Gz(2:N-1,:)./pz(2:N-1,:)));
        
        s = size(R);
        for i = 2:N-1
            for J = 1:s(2)
                if R(i,J)^2 + (Bh/pz(i,J))^2 - 2.*R(i,J).*Mah.*Gz(i,J)/pz(i,J) < 0
                        Psip(i,J) = R(i,J);
                        Psim(i,J) = R(i,J);
                elseif 2*Mah*Gz(i,J)/(R(i,J)*pz(i,J)) > 1
                    Psip(i,J) = Bh./modpz(i,J) + real(sqrt(R(i,J).^2 + (Bh./pz(i,J)).^2 - 2.*R(i,J).*Mah.*Gz(i,J)./pz(i,J)));
                    Psim(i,J) = Bh./modpz(i,J) - real(sqrt(R(i,J).^2 + (Bh./pz(i,J)).^2 - 2.*R(i,J).*Mah.*Gz(i,J)./pz(i,J)));
                end
            end
        end
        Psip = max(R,min(Psip,1-Ymin));     %Regularisation
        Psim = max(R,min(Psim,1-Ymin));     %Regularisation
        
        %Define flux function Q and surface velocity ws
        F1 = Psim.^4 - 4.*(R.*Psim).^2 + (R.^4).*(3-4.*log(R./Psim)) - Psip.^4 + 4.*(R.*Psip).^2 - 4.*R.^2 - 4.*(R.^4).*log(Psip) + 1;
        F2 = 2.*(R.^2).*log(Psip) + Psim.^2 - R.^2 - Psip.^2 + 1 + 2.*(R.^2).*log(R./Psim);
        F3 = Psip.^3 - 3.*Psip.*R.^2 + 3.*R.^2 + 2.*R.^3 + Psim.^3 - 3.*Psim.*R.^2 - 1;
        Q2 = -pz.*F1./16 - R.*Mah.*Gz.*F2./4 - Bh.*sgnpz.*F3./6;
        Q = R.*0;
        Q(2:N-1,:) = Q2(2:N-1,:);
        
        ws = 0.*R;
        G1 = R.^2 - Psim.^2 - 2.*(R.^2).*log(R./Psim) + Psip.^2 - 1 -2.*(R.^2).*log(Psip);
        G2 = log(R.*Psip./Psim);
        G3 = Psip-1-R+Psim;
        ws2 = pz.*G1./4 + R.*Gz.*Mah.*G2 - Bh.*sgnpz.*G3;
        ws(2:N-1,:) = ws2(2:N-1,:);
        
        for i = 2:N-1
            for J = 1:s(2)
                if 2*Mah*Gz(i,J)/(R(i,J)*pz(i,J)) > 1 %&& 1 + (Bh^2)/((pz(i,J)*R(i,J))^2) > 2*Mah*Gz(i,J)/(R(i,J)*pz(i,J))
                    Q(i,J) = Q(i,J) + Bh.*sgnpz(i,J).*(Psim(i,J)^3 + 2*R(i,J)^3 - 3*Psim(i,J)*(R(i,J)^2))./3;
                    ws(i,J) = ws(i,J) -2*Bh*sgnpz(i,J)*(R(i,J) - Psim(i,J));
                end
            end
        end
        
        %Derivative definition to define the ODE system
        dR2 = [2.*(Dzq*Q); -Dzq*(ws.*RG)];
        
    end

    function [Hmax0, isterminal, direction] = EventFn(t,R2)
    %Event function for when max(H) = Hmax. 
    %When this event becomes true the simulation is stopped.
        Hmax0 = (1-Hmax)^2-R2(N);
        isterminal = 1;
        direction = 0;
    end

end