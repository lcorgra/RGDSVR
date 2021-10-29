function [parNet,parFit,parDat,Rtarget]=parametersDeepDecoder(parNet,Rtarget,A,debug)

%PARAMETERSDEEPDECODER   Establishes the parameters of the DD
%   [PARNET,PARFIT,PARDAT,RTARGET]=PARAMETERSDEEPDECODER(PARNET,RTARGET,A,{DEBUG})
%   * PARNET is a structure with network parameters
%   * RTARGET is the target compression ratio
%   * A is a structure with an encoding operation
%   * {DEBUG} serves to show debug info, it defaults to 0
%   ** PARNET is a structure with updated network parameters
%   ** PARFIT is a structure with fitting parameters
%   ** PARDAT is a structure with input data parameters
%   ** RTARGET is the target compression ratio
%

if nargin<4 || isempty(debug);debug=0;end

parNet.useSiren=0;
parFit.net_type=parNet.useSiren;%Network 0 -> deep decoder, network 1 -> siren

%NETTWORK PARAMETERS
parNet.Kred=1;
parNet.use_bias_atac=0;
parNet.wav_L=0;
ND=length(parNet.N);
if ~parNet.useSiren
    parNet.NCo=1;
    parNet.P=parNet.N;
    parNet.wav_typ='db1';
    parNet.wav_over=1;%Overdecoding    
    if parNet.wav_over==0 && parNet.L>0;parNet.P=parNet.N/(2^parNet.wav_L);
    else parNet.P=parNet.N;
    end
    parNet.fact_flatten=1;
    if ND==2;parNet.fact_k=2^(1/parNet.fact_flatten);
    else parNet.fact_k=3^(1/parNet.fact_flatten);%1.2 gives good results
    end
    parNet.use_bias=0;
    parNet.upsample_first=1;%To convolve then upsample   
    parNet.use_sinc=0;%To use sinc instead of linear upsampling
    parNet.use_conv_bn=0;%To use conventional BN-A instead of A-BN
    parNet.use_act=0;%0 relu, 1 swish, 2 sine, 3 tanh, 4 atac
    parNet.use_sinc=1;%To use sinc instead of linear upsampling
    parNet.use_conv_bn=1;%To use conventional BN-A instead of A-BN   
    parNet.use_act=1;%0 relu, 1 swish, 2 sine, 3 tanh, 4 atac
    parNet.use_dct=1;
end
if parNet.useSiren
    parNet.fact_flatten=1;
    parNet.fact_k=1^(1/parNet.fact_flatten);
    parNet.mapping_shape='lapl';
    parNet.mapping_scale=[2 1];
    parNet.train_features=1;
    parNet.use_bias=1;
    parNet.use_conv_bn=-1;
    parNet.use_lowres=0;%Low resolution factor
    parFit.O=-1;%Overlap
end

%FITTING PARAMETERS
if parNet.wav_L>0; parFit.sep_pass=1;else parFit.sep_pass=0;end%Whether to use separable fit
parFit.sep_pass=0;
parFit.epoch_detail=0;%Epoch to start estimation of detail coefficients
parFit.use_prof=0;%To profile
parFit.use_autocast=0;%To cast to half float when possible
parFit.file_name_in='';
parFit.file_name_ou=sprintf('/home/lcg13/Data/DeepDecoder/exp.pt');
if debug;parFit.file_name_ou=sprintf('/home/lcg13/Data/DeepDecoder/exp-debug.pt');end
parFit.lr=0.005;%Learning rate
if debug
    parFit.epochs=1250;%20000;%Number of epochs
    parFit.verbose_frequency=50;%500;%50;%50;%Higher than 1e6 it does not show anything
else
    parFit.epochs=10000;%10000;%5000;
    parFit.verbose_frequency=50;%1e6;%Higher than 1e6 it does not show anything    
end
if parNet.useSiren;parFit.batchGroups=32;parFit.batch_size=prod(parNet.N)/parFit.batchGroups;
else parFit.batch_size=16;
end
parFit.random_input_decay_factor=0.85^(1/(parFit.epochs/40));%Factor to scale the input noise level each decay period---It was 0.85 but worse results
parFit.random_input_sigma=0.005;%Level of random noise to be added to the input
parFit.str_enc.A=A;
parFit.typ_enc='unit';
parFit.factor_best=1;%Factor of improvement for picking best result

%INPUT DATA PARAMETERS
parDat.typ_input=[0 1];%Type of input

