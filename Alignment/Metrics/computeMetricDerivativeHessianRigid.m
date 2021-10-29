function [E,dE,ddE]=computeMetricDerivativeHessianRigid(x,y,metric,H,M,J)

%COMPUTEMETRICDERIVATIVEHESSIANRIGID   Computes a the cost, derivative and
%approximation to the Hessian for a least squares cost |MH(Ex-y)|_2^2 (MSD)
%or |MHEx/|MHEx|-y/|y||_2^2 (NXC) in rigid registration
%   [E,DE,DDE]=COMPUTEMETRICDERIVATIVEHESSIANRIGID(X,Y,{METRIC},{H},{M},{J})
%   * X is the modeled data Ex
%   * Y is the observed data
%   * {METRIC} is the least squares metric, one of the following MSD
%   (default) and NXC 
%   * {H} is a filter (could be extended to other mappings)
%   * {M} is a mask
%   * {J} is the baseline Jacobian dEx
%   ** E is the cost
%   ** DE is the derivative with respect to the rigid motion parameters
%   ** DDE is the approximation to the Hessian with respect to the rigid
%   motion parameters
%

if nargin<3 || isempty(metric);metric='MSD';end
if nargin<4;H=[];end
if nargin<5;M=[];end
if nargin<6;J=[];end

NG=6;
a=[1 2 3 1 1 2 1 2 3 1 2 3 1 2 3 4 4 5 4 5 6;
   1 2 3 2 3 3 4 4 4 5 5 5 6 6 6 5 6 6 4 5 6];
NH=size(a,2);
dE=zeros([NG 1],'single');ddE=zeros([NH 1],'single');

gpu=useGPU;

if gpu;x=gpuArray(x);y=gpuArray(y);end
x=metricFiltering(x,H);y=metricFiltering(y,H);
x=metricMasking(x,M);y=metricMasking(y,M);
x=x(:);y=y(:);
if strcmp(metric,'MSD');[E,ry]=msdMetric(x,y);       
elseif strcmp(metric,'NXC');[E,ry]=nxcMetric(x,y);xn=sqrt(x'*x);x=x/xn;
else error('Metric %s not contemplated\n',metric);
end

if nargout>=2
    if isempty(J);error('No Jacobian provided');end
    for m=1:NG
        if gpu;J{m}=gpuArray(J{m});end
        J{m}=metricFiltering(J{m},H);
        J{m}=metricMasking(J{m},M);
        J{m}=J{m}(:);
        if strcmp(metric,'NXC')%Correction of Jacobian
            J{m}=J{m}/xn;
            J{m}=J{m}-x*(x'*J{m});
        end
    end   
    for m=1:NG;dE(m)=real(gather(J{m}'*ry));end
    if nargout>=3    
        for m=1:NH;ddE(m)=real(gather(J{a(1,m)}'*J{a(2,m)}));end
    end
end
