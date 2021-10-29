function [phi,vt,J,vtg]=precomputeFactorsElasticTransform(T,sp,NG,L,G,ups)

%PRECOMPUTEFACTORSELASTICTRANSFORM   Generates the arrays to apply an
%elastic transform at a given resolution and to transport the gradients
%   [PHI,J,VT,VTG]=PRECOMPUTEFACTORSELASTICTRANSFORM(T,SP,NG,L,G,UPS)
%   * T are the motion parameters (Fourier domain if ups~=0, otherwise 
%   spatial domain)
%   * SP is the spacing of the spatial grid
%   * {NG} is the number of temporal instants for geodesic shooting,
%   defaults to 10
%   * {L} is a symmetric positive-definite differential operator
%   * {G} is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions%   * {G} is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * {UPS} is a padding factor for circular convolutions
%   ** PHI is the transformation (space domain)
%   ** VT is the evolution of the velocity field
%   ** J is the Jacobian of the transformation
%   ** VTG is the spatial gradient of the evolution of the velocity field
% 

gpu=isa(T,'gpuArray');
N=size(T);N(end+1:6)=1;
ND=length(sp);

if nargin<3 || isempty(NG);NG=10;end
if nargin<4 || isempty(L);L=ones(N(1:ND),'like',T);end
if nargin<5 || isempty(G);G='1stFiniteDiscreteCentered';end
if nargin<6 || isempty(ups);ups=2;end

if isscalar(ups);ups=ups*ones(1,ND);end

L=buildDifferentialOperator(L,N,sp,gpu);
if ~any(ups==0);G=buildGradientOperator(G,N,sp,gpu);
else G=[];
end

if nargout>=4;[vt,vtg]=geodesicShooting(T,sp,NG,L,G,ups);else vt=geodesicShooting(T,sp,NG,L,G,ups);end

phi=integrateVelocityFields(vt,sp,G,ups);

%%MAP TO SPACE
N=size(phi);N=N(1:ND);
if ~any(ups==0)
    for m=1:ND;phi=ifftGPU(phi,m)*N(m);end
    phi=real(phi);
end

if nargout>=3;J=computeJacobian(phi,sp,N,ups);end
