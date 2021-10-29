function x=det3x3m(x)

%DET3x3M   Computes the determinant for a set of 3x3 matrices
%   X=DET3x3M(X)
%   * X is the set of input matrices
%   ** X is the set of output determinants
%

N=size(x);
assert(N(1)==N(2) && N(1)==3,'Not 3x3 matrices',N(1),N(2));%Probably also possible to compute for non-square matrices
x11=dynInd(x,[1 1],1:2);
x12=dynInd(x,[1 2],1:2);
x13=dynInd(x,[1 3],1:2);
x21=dynInd(x,[2 1],1:2);
x22=dynInd(x,[2 2],1:2);
x23=dynInd(x,[2 3],1:2);
x31=dynInd(x,[3 1],1:2);
x32=dynInd(x,[3 2],1:2);
x33=dynInd(x,[3 3],1:2);

x=x11.*x22.*x33+x12.*x23.*x31+x13.*x21.*x32-x33.*x12.*x21-x22.*x13.*x31-x11.*x23.*x32;

%Alternative slowly
%x=sum(prod(shearing(x,1,[1 2]),2),1)-sum(prod(shearing(x,-1,[1 2]),2),1);
%This is an alternative but it is much slower
%x=sum(prod(shearing(x,1,[2 1]),1),2)-sum(prod(shearing(x,-1,[2 1]),1),2);