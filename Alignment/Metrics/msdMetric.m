function [rho,r]=msdMetric(x,y,di)

%MSDMETRIC   Computes the mean squared distance between a pair of arrays
%   [RHO,R]=MSDMETRIC(X,Y,{DI})
%   * X is an array
%   * Y is another array
%   * DI are the dimensions on which to compute the metric
%   ** RHO is the mean squared distance
%   ** R are the residuals
%

ND=numDims(x);
if nargin<3 || isempty(di);ga=1;di=1:ND;else ga=0;end

r=x-y;
rho=multDimSum(r.^2,di);
if ga;rho=gather(rho);end