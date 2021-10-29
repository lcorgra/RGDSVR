function rgdsvr_example(idSt,pathAnaconda,pathData)

%RGDSVR_EXAMPLE    Runs an example reconstruction corresponding to Fig.
%II of the manuscript "Fetal MRI by robust deep generative prior
%reconstruction and diffeomorphic registration: application to gestational 
%age prediction," L Cordero-Grande, JE Ortuño-Fisac, A Uus, M Deprez, A 
%Santos, JV Hajnal, and MJ Ledesma-Carbayo, arXiv, 2021.
%   RGDSVR_EXAMPLE({IDST},{PATHANACONDA},{PATHDATA})
%   * {IDST} is the study identifier, it defaults to 34, the case provided
%   as example
%   * {PATHANACONDA} is a path to an anaconda install
%   * {PATHDATA} is a path to the data
%

addpath(genpath('.'));%Add current folder and subfolders to matlab path

if nargin<1 || isempty(idSt);idSt=34;end%Default study
if nargin<2 || isempty(pathAnaconda);pathAnaconda='/home/lcg13/Software/anaconda3';end%Modify with path to anaconda install or include it as input argument
if nargin<3 || isempty(pathData);pathData='../RGDSVR-Data';end%Path with data, modify if required, or include it as input argument

stu=sprintf('%03d',idSt);
environmentName='DeepFetalEnvironment';%Python environment, see instructions in loadPythonDeepFetal.m
loadPythonDeepFetal;%Loads the python code
if gpuDeviceCount;gpuDevice(gpuDeviceCount);end%To use last GPU for matlab

%LOADING A STRUCTURE WITH THE INPUT DATA
%* svr.z is a cell contaning the data for the different stacks
%* svr.MSZ is a cell containing the spacing for the different stacks
%* svr.MTZ is a cell containing a homogeneous matrix encoding the 
%transformation from pixel to physical coordinates for each stack
%* svr.ParZ is a cell containing the slice thickness (field SlZ),
%orientation (field orientation) and slice ordering (field SlOr) for each
%stack
%* svr.R is an array containing a mask of the whole uterus used as the
%reconstruction FOV
%* svr.MSR contains the spacing for the uterus mask
%* svr.MTR contains a homogeneous matrix encoding the transformation from
%pixel to physical coordinates for the mask
load(fullfile(pathData,sprintf('svr_inp_%s',stu)));

%NOTE ABOUT THE COMMENTS: MEANING OF INTERLEAVES AND PACKAGES IN THE CODE
%IS THE OPPOSITE TO THAT IN THE PAPER, I.E., A PACKAGE IN THE CODE IS AN
%INTERLEAVE IN THE PAPER AND VICE VERSA

%SET UP SVR
fprintf('\nSetting up SVR\n');tsta=tic; 
svr=svrAlgorithm(svr);
svr.df=df;%Assign python structure
svr=svrSetUp(svr);        
if isempty(svr);fprintf('Inappropriate slice order identified, SKIPPING\n');return;end
tend=toc(tsta);fprintf('Time setting up: %.3f s\n\n',tend);

%RUN SVR
fprintf('Solving SVR %.2fmm\n',svr.Alg.MS);tsta=tic;
svr=svrAlternateMinimization(svr);
tend=toc(tsta);fprintf('Time solving: %.3f s\n\n',tend);

%WRITE IMAGE
writenii(fullfile(pathData,sprintf('x_%s',stu)),svr.x,[],svr.MSX,svr.MTXWrite)

%SAVE FULL RESULTS
save(fullfile(pathData,sprintf('svr_out_%s',stu)),'svr','-v7.3');
