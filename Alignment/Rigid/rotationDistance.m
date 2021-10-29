function d=rotationDistance(R1,R2,typ)

%ROTATIONDISTANCE  Computes the distance between two rotations given as
%rotation matrices according to 5th and 6th distances in [1] Du Q. Huynh, 
%"Metrics for 3D Rotations: Comparison and Analysis," J Math Imaging Vis 
%35:155-64, 2009. We choose 6th distance, 'GeodesicUnitSphere' by default 
%because it is geometrically more meaningful
%   D=ROTATIONDISTANCE(R1,{R2})
%   * R1 is a rotation matrix or a set of rotation matrices
%   * {R2} is another rotation or set of rotation matrices (it defaults to 
%   identity)
%   * {TYP} is the metric type, possibilities are 'GeodesicUnitSphere'
%   (default), 'DeviationFromIdentity', 'QuickInnerProductQuaternions', 
%   'InnerProductQuaternions', 'NormDifferenceQuaternions' and 
%   'EuclideanDistanceEulerAngles'
%   ** D are the distances between rotations
%

if nargin<2 || isempty(R2);R2=eye(3,'like',R1);end
if nargin<3 || isempty(typ);typ='GeodesicUnitSphere';end

R=matfun(@mtimes,R1,matfun(@ctranspose,R2));
I=eye(3,'like',R1);
if strcmp(typ,'GeodesicUnitSphere')
    N=size(R);N(end+1:3)=1;
    [R,M]=resSub(R,3:numDims(R));M(end+1:3)=1;
    d=zeros(1,M(3));
    for m=1:M(3)
        %eu=wrapToPi(convertRotation(SpinCalc('DCMToEV',R(:,:,m)),'deg','rad'));             
        eu=wrapToPi(rotm2axang(R(:,:,m)));
        d(m)=abs(eu(4));%From -pi to pi, we could normalize to from -1 to 1, we could also take the sin
    end
    d=reshape(d,[ones(1,2) N(3:end)]);
elseif strcmp(typ,'DeviationFromIdentity')
    d=bsxfun(@minus,I,R);
    d=sqrt(multDimSum(abs(d).^2,1:2));
elseif strcmp(typ,'InnerProductQuaternions') || strcmp(typ,'QuickInnerProductQuaternions') || strcmp(typ,'NormDifferenceQuaternions')
    R={R1,R2};
    for n=1:length(R)
        N=size(R{n});N(end+1:3)=1;
        [R{n},M]=resSub(R{n},3:numDims(R{n}));M(end+1:3)=1;
        q{n}=zeros([1 4 M(3)]);
        for m=1:M(3);q{n}(1,:,m)=SpinCalc('DCMToQ',R{n}(:,:,m));end
        q{n}=reshape(q{n},[1 4 N(3:end)]);
    end
    if strcmp(typ,'NormDifferenceQuaternions')
        r{1}=bsxfun(@minus,q{1},q{2});r{2}=bsxfun(@plus,q{1},q{2});
        for n=1:length(r);r{n}=sqrt(sum(abs(r{n}).^2,2));end
        d=min(r{1},r{2});
    else    
        d=abs(sum(bsxfun(@times,q{1},q{2}),2));
        if strcmp(typ,'InnerProductQuaternions');d=acos(d);else d=1-d;end
    end
elseif strcmp(typ,'EuclideanDistanceEulerAngles')%Note this may not be unique as it depends on Euler angle representation...
    R={R1,R2};
    for n=1:length(R)
        N=size(R{n});N(end+1:3)=1;
        [R{n},M]=resSub(R{n},3:numDims(R{n}));M(end+1:3)=1;
        q{n}=zeros([1 3 M(3)]);
        for m=1:M(3);q{n}(1,:,m)=SpinCalc('DCMToEA321',R{n}(:,:,m));end
        q{n}=reshape(q{n},[1 3 N(3:end)]);
    end
    d=sqrt(sum(bsxfun(@minus,q{1},q{2}).^2,2));
else
    error('Distance %s not defined',typ);
end

    