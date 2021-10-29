function [gpu,tD,NT,dH,Tupd,E,NXX,spGridX,NXT,spGridT,ups,Fx,FHx,lambda,NPr,flagw,NcontTest,winit]=initializeDEstimation(DT,NX,MSS,NXT,ups,lambda,NconvT,winit)

%INITIALIZEDESTIMATION   Initializes variables used generically of deformable motion estimation in SVR 
%   [GPU,TD,NT,DH,TUPD,E,NXX,SPGRIDX,NXT,SPGRIDT,UPS,FX,FHX,LAMBDA,NPR,FLAGW,NCONTTEST,WINIT]=INITIALIZEDESTIMATION(DT,NX,MSS,NXT,UPS,LAMBDA,NCONVT,WINIT)
%   * DT is the starting motion transform
%   * NX is the spatial grid size
%   * MSS is the spatial grid resolution
%   * NXT is the temporal grid size
%   * UPS are the upsampling factors
%   * LAMBDA is the regularization weight
%   * NCONVT is the required number of iterations for resetting convergence
%   plus 1
%   * WINIT is the starting parameter for deformable registration
%   ** GPU is a flag for gpu computations
%   ** TD is the time dimension
%   ** NT is the number of transforms
%   ** DH is the gradient update
%   ** TUPD are candidate velocity updates
%   ** E is the energy
%   ** NXX is the spatial grid size
%   ** SPGRIDX is the spatial grid resolution
%   ** NXT is the temporal grid size
%   ** SPGRIDT is the temporal grid resolution
%   ** UPS are the upsampling factors
%   ** FX serves for domain mapping in image grid
%   ** FHX serves for domain mapping in image grid
%   ** LAMBDA is the regularization weight
%   ** NPR is the number of paramters for reshaping
%   ** FLAGW are flags used to indicate partial convergence for different
%   motion states
%   ** NCONTEST is the required number of iterations for resetting
%   convergence
%   ** WINIT is the starting parameter for deformable registration
%

gpu=useGPU;
pD=4;%Dimension of parameters
tD=5;%Dimension of transforms
NT=size(DT,tD);%Number of transforms
NP=size(DT,pD)*prod(NXT);%Number of parameters
dH=zeros([NP NT],'like',DT);
Tupd=zeros(size(DT),'like',DT);
E=zeros([1 NT],'single');
NXX=NX;
spGridX=MSS;
spGridT=spGridX.*(NXX./NXT);
Fx=buildMapSpace(dynInd(DT,1,tD),1,round(NXT*ups(1)),NXT);
FHx=buildMapSpace(dynInd(DT,1,tD),0,round(NXT*ups(1)),NXT);
NPr=size(DT);%Number of parameters for reshaping
flagw=zeros([1 NT],'single');
NcontTest=NconvT-1;
if ups(1)==0;ups(2)=0;end%Computations will be performed in space
