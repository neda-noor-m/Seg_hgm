function holistic(DATASET_NAME, cfg)

%input: 

%DATASET_NAME could be 'msrc' or 'voc'
%modelsuffix is the name of the result folder
% cfg is the configuration file

if nargin < 2
    cfg = 'msrc_full';
end;
modelsuffix = ['-' cfg];
overwrite = 0;

dataset_globals;
[featparams, learnparams, infer_params] = feval(cfg);

featparams = defaultfeatparams(featparams);
detclasses = featparams.det_classes;

boxdirname = BBOX_NAME;
[classes, ~] = getclassinfo(DATASET_NAME); 
classes = filterclasses(classes, IGNORE_CLASSES); 

 % which classes (ind_classes) are to be used in detection
if isnumeric(detclasses) 
    ind_classes = find_ind_class(detclasses,IGNORE_CLASSES);
else
    ind_classes = zeros(length(detclasses),1);
    for cls = 1 : length(detclasses)
        i_class = find_ind_class(classes, detclasses{cls});
        ind_classes(cls) = i_class;
    end
end

featparams.ind_classes = ind_classes;
featparams.ALLPATHS = ALLPATHS;
featparams = defaultfeatparams(featparams, ind_classes);
learnparams.featparams = featparams;

% the name of the run -- will be stored in directory
modelname = ['M' num2str(length(detclasses)) modelsuffix];

if overwrite
    learnpath = fullfile(ALLPATHS.LEARNING_PATH, modelname);
    if exist(learnpath, 'dir')
       unix(sprintf('rm -fR %s', learnpath));
    end;
    resultspath = fullfile(ALLPATHS.RESULTS_PATH, modelname);
    if exist(resultspath, 'dir')
       unix(sprintf('rm -fR %s', resultspath));
    end;
end;
    

% if you want to train model on only a subset of images, set ind_images =
% subset of images; Otherwise either ind_images = [] or don't pass it to
% learning_segmentation()
%EXAMPLE:
%ind_images = [20]; Learning code will run on only the 20th image from training
%ind_images = []; Run on all the images from the training dataset

ind_images = [];

% TRAINING
disp('-----TRAINING the model-----')
learning_segmentation('k1', K1, 'k2', K2, 'modelname', modelname, 'ind_classes', ind_classes,...
                              'ind_images', ind_images, 'boxdirname', boxdirname, 'learnparams', learnparams);

% INFERENCE
whichtest = 'test';  % let's run it on the same image as we trained on
                      % for testing set whichtest = 'test'
ind_images = [];      % again leave this empty if you want all images processed  

suffix = sprintf('eps-%0.1f', infer_params.eps);
disp('-----INFERENCE-----')
printout=0;
inference_segmentation(modelname, whichtest, ind_images, infer_params, suffix, DATASET_NAME, printout)

% COMPUTE ACCURACY
try   
    rPATH = fullfile(ALLPATHS.RESULTS_PATH, modelname, suffix, 'seg');
    ucmpath = ALLPATHS.UCM1_PATH;
    if featparams.segOnly
        ucmpath = ALLPATHS.UCM2_PATH;
    end;
    disp('-----PERFORMANCE-----')
    [acc average globalAccuracy] = pixelwiseAccuracy(ALLPATHS.GT_PATH, rPATH, ucmpath, modelname, DATASET_NAME, featparams, ind_images);
catch
    fprintf('ERROR IN COMPUTING ACCURACY!\n');
end;