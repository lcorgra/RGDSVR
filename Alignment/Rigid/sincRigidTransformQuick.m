function [x,xB]=sincRigidTransformQuick(x,et,di,F,FH,gat)

%SINCRIGIDTRANSFORMQUICK rigidly transforms volumes by sinc-based 
%   interpolation (both forwards and backwards) using 4 shears as described
%   in [1] JS Welling, WF Eddy, TK Young, "Rotation of 3D volumes by 
%   Fourier-interpolated shears," Graph Mod 68:356-370, 2006
%   [X,XB]=SINCRIGIDTRANSFORMQUICK(X,ET,{DI},{F},{FH},{GAT},{NR})
%   * X is a volume
%   * ET are the transform factors
%   * {DI} is a flag to indicate whether to perform direct (default) or 
%   inverse transform
%   * {F} contains discrete Fourier transform matrices
%   * {FH} contains inverse discrete Fourier transform matrices
%   * {GAT} serves to gather the surrogate information
%   ** X is the rigidly transformed volume
%   ** XB are intermediate results that can be reused to compute the 
%   gradient of the transform
%

if isempty(et);xB=[];return;end

if nargin<3 || isempty(di);di=1;end
if nargin<4 || isempty(F);F={[],[],[]};end
if nargin<5 || isempty(FH);FH={F{1}',F{2}',F{3}'};end
if nargin<6 || isempty(gat);gat=0;end

NX=size(x);NX(end+1:3)=1;NX=NX(1:3);
if ~isempty(F{1});NR=[size(F{1},1) size(F{2},1) size(F{3},1)];else NR=NX;end

if any(NR~=NX);x=resampling(x,NR,3);end

tr=[2 1 3 2;
    3 2 1 3;
    1 3 2 1
    2 3 1 2;
    3 1 2 3;
    1 2 3 1];%Set of shears
tr=tr(real(et{2}),:);

mV=1:4;
if di==0;mV=flip(mV);end

if di
    for m=1:3     
        if et{5}(m)==1;x=flipping(x,et{4}{m});end%FLIPPING FOR LARGER THAN 90DEG ROTATIONS
    end
end

if nargout>=2;xB=cell(1,4);end

for m=mV
    x=fftGPU(x,tr(m),F{tr(m)});     
    x=bsxfun(@times,x,et{1}{m});
    if nargout>=2
        xB{m}=x;
        if gat;xB{m}=gather(xB{m});end
    end
    x=ifftGPU(x,tr(m),FH{tr(m)});
end

if ~di
    for m=1:3
        if et{5}(m)==1;x=flipping(x,et{4}{m});end%FLIPPING FOR LARGER THAN 90DEG ROTATIONS
    end
end

if any(NR~=NX);x=resampling(x,NX,3);end
