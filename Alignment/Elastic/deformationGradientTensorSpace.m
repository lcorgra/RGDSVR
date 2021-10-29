function ud=deformationGradientTensorSpace(u,sp)

%DEFORMATIONGRADIENTTENSORSPACE   Computes the gradient/divergence of a 
%deformation field, uses 1stFiniteDiscreteCentered
%   UD=DEFORMATIONGRADIENTTENSORSPACE(U,SP)
%   * U is the deformation field
%   * SP is the spacing of the spatial grid
%   ** UD is the gradient
% 

ND=length(sp);
gpu=isa(u,'gpuArray');
M=size(u);M(end+1:ND)=1;M=M(1:ND);
N=size(u);N(end+1:ND)=1;
sp=2*sp;

if numDims(u)==ND+2%We compute the divergence
    ud=zeros(N(1:ND+1),'like',u);        
    for n=1:ND
        uu=dynInd(u,n,ND+2);
        ud=ud+(dynInd(uu,[2:M(n) 1],n)-dynInd(uu,[M(n) 1:M(n)-1],n))/sp(n);
    end
elseif numDims(u)==ND+1%We compute the deformation tensor of a field
    ud=zeros([N ND],'like',u);
    for n=1:ND;ud=dynInd(ud,n,ND+2,(dynInd(u,[2:M(n) 1],n)-dynInd(u,[M(n) 1:M(n)-1],n))/sp(n));end
else%We compute the gradient    
    N(end+1:ND+3)=1;
    ud=zeros([N(1:ND) ND N(ND+2:ND+3)],'like',u);
    for n=1:ND;ud=dynInd(ud,n,ND+1,(dynInd(u,[2:M(n) 1],n)-dynInd(u,[M(n) 1:M(n)-1],n))/sp(n));end
end
