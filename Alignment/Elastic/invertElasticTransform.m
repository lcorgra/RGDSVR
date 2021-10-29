function [v,n,Dg,vold,Dgmax]=invertElasticTransform(u,sp,ups,checkConvergence,thInvert,tolU,nIt,rGrid,interpMethod,Npad)
 
%INVERTELASTICTRANSFORM   Inverts a transformation following the formulation
%in M Chen et al, "A simple fixed-point approach to invert a deformation 
%field," Med Phys, 35(1):81-88, Jan 2008
%   [V,VOLD,N]=INVERTELASTICTRANSFORM(U,SP,{UPS},{CHECKCONVERGENCE},{THINVERT},{TOLU},{NIT},{INTERPMETHOD},{NPAD})
%   * U is the forward displacement field
%   * SP is the spacing
%   * {UPS} is an upsampling factor to interpolate
%   * {CHECKCONVERGENCE} serves to check convergence only
%   * {THINVERT} is the threshold for invertibility
%   * {TOLU} is the tolerance
%   * {NIT} is the maximum number of iterations
%   * {INTERPMETHOD} is the interpolation method
%   * {NPAD} indicates whether to use padding for periodic boundary 
%conditions (if a 1) or the padding values (if a vector)
%   ** V is the inverse displacement field
%   ** N are the iterations till convergence
%   ** DG is a spatial map of Lipschitz constants
%   ** VOLD are the residuals
% 

if nargin<3 || isempty(ups);ups=1;end
if nargin<4 || isempty(checkConvergence);checkConvergence=0;end
if nargin<5 || isempty(thInvert);thInvert=0.85;end
if nargin<6 || isempty(tolU);tolU=1e-2;end
if nargin<7 || isempty(nIt);nIt=100;end
if nargin<8 || isempty(interpMethod);interpMethod='linear';end
if nargin<9 || isempty(Npad);Npad=1;end

ND=length(sp);
N=size(u);N=N(1:ND);
if isscalar(ups);ups=ups*ones(1,ND);end
gpu=isa(u,'gpuArray');
M=round(N.*ups);ups=M./N;
sp=sp./ups;

u=resampling(u,M);
%Check convergence
if checkConvergence || nargout>=3
    if ND==2
        neigh=[1  0;
               0  1;           
               1  1;          
              -1  1];
    else
       neigh=[0  0  1;
               0  1  0;
               1  0  0;
               1  1  0;
               0  1  1;
               1  0  1;
              -1  1  0;
               0 -1  1;
               1  0 -1];
    end
    Dg=zeros(M(1:ND),'like',u);    
    for n=1:size(neigh,1)        
        ud=sum((u-circshift(u,[neigh(n,:) 0])).^2,ND+1)/sum(bsxfun(@times,neigh(n,:),sp).^2);
        Dg=max(Dg,ud);
        if nargout>=3
            ud=circshift(ud,-[neigh(n,:) 0]);
            Dg=max(Dg,ud);
        end      
    end
    Dg=sqrt(Dg);Dgmax=max(Dg(:));
    %fprintf('Maximum value: %.2f\n',max(Dg(:)));
    if checkConvergence && Dgmax>=thInvert;v=0;else v=1;end
    if nargout>=3;Dg=resampling(Dg,N);end
    if checkConvergence;vold=[];n=[];return;end    
end

for n=1:ND;rGrid{n}=single(1:M(n))*sp(n);end
extr=0;
if isscalar(Npad) && Npad==1%Extrapolate
    extr=2;
    ua=abs(u);
    Npad=zeros(1,ND);
    for n=1:ND
        Npad(n)=2*(ceil(gather(multDimMax(dynInd(ua,{[1 M(n)],n},[n ND+1]),1:ND)))+1);
        Npad(n)=min(Npad(n),M(n));
        rG=(1:Npad(n))*sp(n);
        rGrid{n}=cat(2,-flip(rG)+rGrid{n}(1),rGrid{n},rGrid{n}(M(n))+rG);               
        %u=cat(n,dynInd(u,M(n)-Npad(n)+1:M(n),n),u,dynInd(u,1:Npad(n),n));        
    end
    u=padarray(u,Npad,'circular','both');
    %fprintf('Padding:%s\n',sprintf(' %d',Npad));
end
if gpu
    for n=1:ND;rGrid{n}=gpuArray(rGrid{n});end
end
if any(Npad(:)~=0) && extr~=2;extr=1;end%extr=1 implies that we pad as suggested in Npad, extr=2 implies that we calculate padding

r=gridv(rGrid);
Mpad=size(r);
ROI=cell(1,ND);
for n=1:ND;ROI{n}=1+Npad(n):Mpad(n)-Npad(n);end

v=zeros(Mpad,'like',u);
for n=1:nIt
    vp=v+r;
    vold=v;
    for s=1:ND
        if ND==2;v=dynInd(v,s,ND+1,-(interpn(dynInd(r,1,ND+1),dynInd(r,2,ND+1),dynInd(u,s,ND+1),dynInd(vp,1,ND+1),dynInd(vp,2,ND+1),interpMethod,0)));
        else v=dynInd(v,s,ND+1,-(interpn(dynInd(r,1,ND+1),dynInd(r,2,ND+1),dynInd(r,3,ND+1),dynInd(u,s,ND+1),dynInd(vp,1,ND+1),dynInd(vp,2,ND+1),dynInd(vp,3,ND+1),interpMethod,0)));
        end
    end   
    vold=v-vold;
    if extr;vold=dynInd(vold,ROI,1:ND);end
    %if sqrt(vold(:)'*vold(:))<tolU*prod(N(1:ND));break;end
    if max(abs(vold(:)))<tolU;break;end%Stricter
end
if n==nIt;fprintf('Inverting the field reached maximum number of iterations without converging. Error: %.2f\n',max(abs(vold(:))));end

if extr==2%If we call with Npad=1 we confine, otherwise we return as it is
    v=dynInd(v,ROI,1:ND);
    v=resampling(v,N);
    if nargout>=4
        vold=vecnorm(vold,2,ND+1);
        vold=resampling(vold,N);
    end    
end
