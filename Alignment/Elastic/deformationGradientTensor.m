function ud=deformationGradientTensor(u,sp,G,fo,N)

%DEFORMATIONGRADIENTTENSOR   Computes the gradient/divergence of a deformation 
%field in the Fourier domain
%   UD=DEFORMATIONGRADIENTTENSOR(U,SP,{G},{FO})
%   * U is the deformation field
%   * SP is the spacing of the spatial grid
%   * {G} is the gradient type, one of the following 
%'1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%'1stFiniteDiscreteCentered' (default) or if a cell, the filters along the 
%different dimensions
%   * {FO} indicates if the field is already in the Fourier domain (1) or
%   Fourier domain shifted (2), it defaults to 0
%   * {N} indicates the resolution to compute the tensor
%   ** UD is the gradient
% 

if nargin<3 || isempty(G);G='1stFiniteDiscreteCentered';end

ND=length(sp);
gpu=isa(u,'gpuArray');
if nargin<4 || isempty(fo);fo=0;end
if nargin<5;N=size(u);N(end+1:ND)=1;end
M=size(u);M(end+1:ND)=1;M=M(1:ND);
u=resampling(u,N,fo);
N=size(u);N(end+1:ND)=1;
sp=sp./(N(1:ND)./M);

G=buildGradientOperator(G,N(1:ND),sp,gpu);
if fo==2
    for n=1:ND;G{n}=fftshift(G{n},n);end
end

if numDims(u)==ND+2%We compute the divergence
    ud=zeros(N(1:ND+1),'like',u);
    if fo>0
        for n=1:ND;ud=ud+bsxfun(@times,dynInd(u,n,ND+2),G{n});end
    else
        for n=1:ND;ud=ud+real(filtering(dynInd(u,n,ND+2),G{n}));end
    end
elseif numDims(u)==ND+1%We compute the deformation tensor of a field
    ud=zeros([N ND],'like',u);
    if fo>0
        for n=1:ND;ud=dynInd(ud,n,ND+2,bsxfun(@times,u,G{n}));end
    else
        for n=1:ND;ud=dynInd(ud,n,ND+2,real(filtering(u,G{n})));end
    end
else%We compute the gradient    
    N(end+1:ND+3)=1;
    ud=zeros([N(1:ND) ND N(ND+2:ND+3)],'like',u);
    if fo>0
        for n=1:ND;ud=dynInd(ud,n,ND+1,bsxfun(@times,u,G{n}));end
    else
        for n=1:ND;ud=dynInd(ud,n,ND+1,real(filtering(u,G{n})));end
    end
end
