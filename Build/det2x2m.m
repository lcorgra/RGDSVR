function x=det2x2m(x)

%DET2x2M   Computes the determinant for a set of 2x2 matrices
%   X=DET2x2M(X)
%   * X is the set of input matrices
%   ** X is the set of output determinants
%

N=size(x);
assert(N(1)==N(2) && N(1)==2,'Not 2x2 matrices',N(1),N(2));%Probably also possible to compute for non-square matrices
x=diff(prod(shearing(x,1,[1 2]),2),1);