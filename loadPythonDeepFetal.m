pythonBin=fullfile(pathAnaconda,'envs',environmentName,'bin','python');
system(sprintf('%s -m pip install Python/deepfetal/',pythonBin));

pe=pyenv;
if pe.Status=='Loaded';terminate(pyenv);end
pyenv('Version',fullfile(pathAnaconda,'envs',environmentName,'bin','python'));pyenv('ExecutionMode','OutOfProcess');
if ~exist('df','var');df=py.importlib.import_module('deepfetal');end
py.importlib.reload(df);

%%%RECOMMENDED INSTRUCTIONS FOR CREATING THE ENVIRONMENT (TESTED IN ANACONDA3)
%conda create -n DeepFetalEnvironment python=3.7
%conda install -n DeepFetalEnvironment pytorch pywavelets torchvision cudatoolkit=11 -c pytorch
