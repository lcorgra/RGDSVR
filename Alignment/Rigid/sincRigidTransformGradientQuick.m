function x=sincRigidTransformGradientQuick(xB,et,etg,F,FH,gat,NX)

%SINCRIGIDTRANSFORMGRADIENTQUICK obtains the gradient of the sinc rigid
%transform of volumes using 4 shears as described in [1] JS Welling, WF 
%Eddy, TK Young, "Rotation of 3D volumes by Fourier-interpolated shears," 
%Graph Mod 68:356-370, 2006
%   G=SINCRIGIDTRANSFORMGRADIENTQUICK(X,ET,{DI},{F},{FH},{GAT},{NX})
%   * XB are the volumes after the first, second, third and fourth Fourier 
%   Transform
%   * ET are the transform factors
%   * ETG are the transform gradient factors
%   * {F} contains discrete Fourier transform matrices
%   * {FH} contains inverse discrete Fourier transform matrices
%   * {GAT} serves to gather the surrogate information
%   * {NX} serves to invert previous padding
%   ** G is the gradient of the transformed image
%

if nargin<4 || isempty(F);F={[],[],[]};end
if nargin<5 || isempty(FH);FH={F{1}',F{2}',F{3}'};end
if nargin<6 || isempty(gat);gat=0;end
NR=size(xB{1});NR(end+1:3)=1;
if nargin<7 || isempty(NX);NX=NR;end

gpu=useGPU;

tr=[2 1 3 2;
    3 2 1 3;
    1 3 2 1
    2 3 1 2;
    3 1 2 3;
    1 2 3 1];%Set of shears
tr=tr(real(et{2}),:);

if gpu;xB{1}=gpuArray(xB{1});end
for n=1:2;x{n}=bsxfun(@times,xB{1},etg{1}{n});end
if gat;xB{1}=gather(xB{1});end
for n=1:2
    x{n}=ifftGPU(x{n},tr(1),FH{tr(1)});
    x{n}=fftGPU(x{n},tr(2),F{tr(2)});
    x{n}=bsxfun(@times,x{n},et{1}{2});
end
if gpu;xB{2}=gpuArray(xB{2});end
for n=3:5;x{n}=bsxfun(@times,xB{2},etg{2}{n-2});end
if gat;xB{2}=gather(xB{2});end
for n=1:5
    x{n}=ifftGPU(x{n},tr(2),FH{tr(2)});
    x{n}=fftGPU(x{n},tr(3),F{tr(3)});
    x{n}=bsxfun(@times,x{n},et{1}{3});
end
if gpu;xB{3}=gpuArray(xB{3});end
for n=6:8;x{n}=bsxfun(@times,xB{3},etg{3}{n-5});end
if gat;xB{3}=gather(xB{3});end
for n=1:8
    x{n}=ifftGPU(x{n},tr(3),FH{tr(3)});
    x{n}=fftGPU(x{n},tr(4),F{tr(4)});
    x{n}=bsxfun(@times,x{n},et{1}{4});
end
if gpu;xB{4}=gpuArray(xB{4});end
for n=9:11;x{n}=bsxfun(@times,xB{4},etg{4}{n-8});end
if gat;xB{4}=gather(xB{4});end
ND=numDims(x{1});
x=cat(ND+1,x{:});
x=aplGPU(etg{5}',x,ND+1);
x=num2cell(x,1:ND);
for m=1:6
    x{m}=ifftGPU(x{m},tr(4),FH{tr(4)});
    if any(NR~=NX);x{m}=resampling(x{m},NX,m);end
    if gat;x{m}=gather(x{m});end
end
