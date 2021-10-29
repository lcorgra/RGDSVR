function R=generatePrincipalAxesRotations(irr)

%GENERATEPRINCIPALAXESROTATIONS  Generates 24(48) possible rotations of
%coordinate systems using permutations and flipping of axes. It inherits
%from the discussion in
%https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array
%   R=GENERATEPRINCIPALAXESROTATIONS(IRR)
%   * {IRR} indicates whether to also generate irregular rotations, it
%   defaults to 0, 1 is all possible rotations and 2 is all without
%   permutations, -1 is all without permutations and irregular
%   ** R are the generated rotations
%

if nargin<1 || isempty(irr);irr=0;end

%Permutation tensor:
A(:,:,1)=[1 0 0;
          0 1 0;
          0 0 1];
if irr<2 && irr>=0
    A(:,:,2)=[0 1 0;
              0 0 1;
              1 0 0];
    A(:,:,3)=[0 0 1;
              1 0 0;
              0 1 0];
end
%Negating terms so the product of the diagonal is 1
B(:,:,1,1)=[1 0 0;
            0 1 0;
            0 0 1];
B(:,:,1,2)=[-1 0 0;
             0 -1 0;
             0 0 1];
B(:,:,1,3)=[-1 0 0;
            0 1 0;
            0 0 -1];
B(:,:,1,4)=[1 0 0;
            0 -1 0;
            0 0 -1];
%Positive or negative permutation
C(:,:,1,1,1)=[1 0 0;
              0 1 0;
              0 0 1];
if irr<2 && irr>=0
    C(:,:,1,1,2)=[0 0 -1;
                  0 -1 0;
                 -1 0 0];
end
%For irregular rotations
D(:,:,1,1,1,1)=[1 0 0;
                0 1 0;
                0 0 1];
if irr>0
    D(:,:,1,1,1,2)=[-1 0 0;
                    0 -1 0;
                    0 0 -1];            
end
[A,B,C,D]=parUnaFun({A,B,C,D},@single);
R=matfun(@mtimes,matfun(@mtimes,A,B),matfun(@mtimes,C,D));
R=resSub(R,3:6);
