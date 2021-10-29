function x=metricFiltering(x,H)

%METRICFILTERING   Computes a filter for a modified metric
%   X=METRICFILTERING(X,H)
%   * X is an array
%   * H is a filter
%   ** X is the filtered data
%

if ~isempty(H)
    if iscell(H)%Separable
        for n=1:length(H);x=filtering(x,H{n});end
    else%Joint
        x=filtering(x,H);
    end
end
