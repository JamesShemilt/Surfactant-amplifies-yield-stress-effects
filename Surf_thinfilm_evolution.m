%Surf_thinfilm_evolution
%March 2023

%Function to used to solve thin-film evolution equations in Shemilt et al. 
%(2023) Surfactant amplifies yield-stress effects in the capillary 
%instability of a film coating a tube. This code is used to get data shown
%in figures 3, 4, 5 and 6. 

%The input parameters are: B, capillary Bingham number; Ma, Marangoni
%number; A, initial perturbation amplitude; Ymin, regularisation parameter;
%N, number of finite difference grid points; Tol, error tolerance in ODE 
%solver; Tend, end time for integration.

function solH = Surf_thinfilm_evolution(B,Ma,A,Ymin,N,Tol,Tend)

k = sqrt(2)/2;
L = pi/k;               %domain length
dz = pi/(k*(N-1));      %grid spacing
z = linspace(0,L,N);    %finite difference grid

%Initial Conditions
    H0 = 1 - A.*cos(k.*z');
    G0 = zeros(N,1) + 1;
    H0G0 = [H0; G0];

% %     Derivative Matrices
%--% Dx and Dxxx enforce symmetry BCs, Dxq enforces an antisymmetry BC
%--% which are used in the evolution equations.
Dz = zeros(N);
Dxx = zeros(N);
Dzzz = zeros(N);
Dzq = zeros(N);

for n = 2:N-1
    Dz(n,n-1) = -1/(2*dz);
    Dz(n,n+1) = 1/(2*dz);
end

for n = 3:N-2
    Dzzz(n,n-2:n+2)= [-0.5, 1, 0,-1,0.5]/(dz^3);
end
Dzzz(2,1:4) = [1, -0.5, -1, 0.5]/(dz^3);
Dzzz(N-1,N-3:N) = [-0.5, 1, 0.5,-1]/(dz^3);

Dzq(1,2) = 1/dz;
for n = 2:N-1
    Dzq(n,n-1) = -1/(2*dz);
    Dzq(n,n+1) = 1/(2*dz);
end
Dzq(N,N-1) = -1/dz;

Dzzz = sparse(Dzzz);
Dz = sparse(Dz);
Dzq = sparse(Dzq);

M = -Dz-Dzzz;

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

% %         Options for ode15s
options = odeset('RelTol',Tol,'AbsTol',Tol, 'Vectorized', 'on', 'JPattern', Sparsity);%,'InitialStep',1e-12);

% %         Integration Time
tspan = [0 Tend];

% %     Solve with ode15s
solH = ode15s(@deriv,tspan,H0G0,options);

    function dH = deriv(~,H)
        
        pz = M*H(1:N,:);        %capillary pressure gradient
        modpz = abs(pz);        %modulus of pressure gradient
        sgnpz = sign(pz);       %sign of pressure gradient
        G = H(N+1:end,:);       %Surfactant concentration
        Gz = Dz*G;              %Concentration gradient
        h = H(1:N,:);           %Layer height
            
        % %         Y-
        Ym = 0*h;
        Ym(2:N-1,:) = H(2:N-1,:) - B./modpz(2:N-1,:)+Ma.*Gz(2:N-1,:)./pz(2:N-1,:);
        Ym = max(Ymin,min(h,Ym));
        
        % %         Y+
        Yp = h;
        Yp(2:N-1,:) = H(2:N-1,:) + B./modpz(2:N-1,:)+Ma.*Gz(2:N-1,:)./pz(2:N-1,:);
        Yp = max(Ymin,min(h,Yp));
       
        % %         Volume Flux
        q = -pz.*(h.^3 + (h-Yp).^3 - (h-Ym).^3)./3;
        q = q -(Ma/2).*Gz.*(h.^2 + (h-Yp).^2 - (h-Ym).^2);
        q = q + (B/2).*sgnpz.*(h.^2 - (h-Yp).^2 - (h-Ym).^2);
        q(1,:) = 0.*q(1,:);
        q(N,:) = 0.*q(N,:);

        % %         Surface Velocity
        ws = -(1/2).*pz.*(h.^2 + (h-Yp).^2 - (h-Ym).^2 ) - B.*sgnpz.*(h-Yp-Ym) - Ma.*Gz.*(h-Yp+Ym);
        ws(1,:) = 0.*ws(1,:);
        ws(N,:) = 0.*ws(N,:);
        
        % %         Derivative definition from evolution equations
        dH = [-Dzq*q; -Dzq*(ws.*G)];
    
    end

end
