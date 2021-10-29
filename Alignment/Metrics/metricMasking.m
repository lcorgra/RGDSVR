function x=metricMasking(x,M,softMask)

%METRICMASKING   Computes a masking for a metric
%   X=METRICMASKING(X,{M},{SOFTMASK})
%   * X is an array
%   * {H} is a mask
%   * {SOFTMASK} indicates whether to use soft masking (default)
%   ** X is the masked data
%

if nargin<2;M=[];end
if nargin<3 || isempty(softMask);softMask=1;end

if ~isempty(M)
    if ~softMask;M=single(M>0);end
    x=bsxfun(@times,x,M);
end
