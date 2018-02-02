function [features, empmean, CRFDATA] = computeFeatures(imlist, NumStates, ignore_classes, ind_classes, gamma, cache_dir, ALLPATHS, train, featparams)
% INPUT
%    imlist         ... list of images for which the features are to be computed
%    NumStates      ... the number of states for superpixel and supersegment
%    ignore_classes ... class indices that are to be ignored (eg, for MSRC
%                       we don't use 'void' (1), 'horse' (6), and
%                       'mountain' (9)
%    ind_classes    ... indices of classes for which we want to compute
%                       detection features for (these indices should be for
%                       the already cleaned class list (see 'ignore classes')
%    cache_dir      ... where to cache the features (important for debugging)
%    ALLPATHS       ... should contain names of directories for all
%                       features (see dataset_globals)
%
% OUTPUT
%    features       ... the features
%    empmean        ... empirical mean

if nargin < 8, train = 1; end; % are we training?
if nargin < 9 || isempty(featparams)
    featparams = defaultfeatparams([], ind_classes);
else
    featparams = defaultfeatparams(featparams, ind_classes);
end;

% set up the paths
BBOX_PATH = ALLPATHS.BBOX_PATH;

% load the detection thresholds
[classes, gtcols] = getclassinfo(ALLPATHS.DATASET_NAME);
if numel(ignore_classes)
   classes = filterclasses(classes, ignore_classes);   % classes
   gtcols(ignore_classes,:) = [];
end;
ALLPATHS.classes = classes;          

thresholds = zeros(length(classes), 1);
for i = 1 : length(ind_classes)
    class_name = classes(ind_classes(i)).class;
    if featparams.uselogistic == 0
        thresh = load(fullfile(BBOX_PATH, 'threshdata', [class_name '.txt']));
    else
        thresh = load(fullfile(BBOX_PATH, 'threshdata', [class_name '_or.txt']));
    end;
    thresholds(ind_classes(i)) = - thresh;
end;
%-------------------------

%-------------------------
% load masks for detector
maskfile = fullfile(BBOX_PATH, 'masks', 'masks.mat');
try
    data = load(maskfile, 'prob_maps');
    masks = data.prob_maps;
catch
    masks = [];
    if ~strcmp(ALLPATHS.DATASET_NAME, 'msrc'), fprintf('ERROR to load the maskfile!\n'); end;
end;

if exist(fullfile(ALLPATHS.Z_FEATURE_PATH,'frequency.mat'),'file')
    F = load(fullfile(ALLPATHS.Z_FEATURE_PATH,'frequency.mat'));
    stats = F.stats;   %unary feature for Z
    freq = F.freq;     %Chowliu feature for Z
else
    [stats, freq, ~] = computeFrequencyOfClasses(ALLPATHS.DATASET_NAME);
end

% COMPUTE THE FEATURES
NumFeatures = getNumFeatures(featparams);   % computing the number of features based on input parameters
features = [];
empmean = zeros(NumFeatures, 1);
vars = [];

% add variables
[vars] = addvar(vars, length(imlist), 'SP'); % n = numel(A) returns the the number of elements, n, in array A.
if featparams.segFeature == 1
    [vars] = addvar(vars, length(imlist), 'SS');
end
[vars] = addvar(vars, length(imlist), 'Z');
for i = 1 : length(ind_classes)
    biname = sprintf('%s-%d', 'bi', ind_classes(i));
    [vars] = addvar(vars, length(imlist), biname);
end;
[vars] = addvar(vars, length(imlist), 'scene');

CRFDATA.COUNT_UNARY = zeros(length(imlist), 1);  % count for variables
CRFDATA.COUNT_FACTORS = zeros(length(imlist),1);   % count variable for the factors
CRFDATA.vars = vars;
CRFDATA.NumClasses = NumStates;
CRFDATA.featparams = featparams;

if numel(cache_dir) && ~exist(cache_dir, 'dir')
    mkdir(cache_dir); 
end
storefeats = 0;

% superpixels feature:
disp('Computing superpixel feature');
featfile = fullfile(cache_dir, 'feat_SP');
if ~featparams.segOnly
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
        tic;
        if train, CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, 'SP'); end;
        [features, empmean, CRFDATA] = computeSPfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train, featparams.GlobalFeatures);
        e=toc;
        fprintf(' (%0.4f sec)\n', e);
        if numel(cache_dir) && storefeats
            save(featfile, 'features', 'empmean', 'CRFDATA'); 
        end
    end
end;
if featparams.spOnly == 1
    empmean = empmean(1);
    return;
end;

% my code
% new superpixels feature: local information per each superpixel + global inf
% if featparams.GlobalFeatures
%     disp('Computing global feature');
%     featfile = fullfile(cache_dir, 'feat_SP');
%     if exist(featfile, 'file') && numel(cache_dir)
%         load(featfile);
%     else
%         tic;
%         if train, CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, 'SP'); end;
%         [features, empmean, CRFDATA] = computeGLBfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train);
%         e=toc;
%         fprintf(' (%0.4f sec)\n', e);
%         if numel(cache_dir) && storefeats
%             save(featfile, 'features', 'empmean', 'CRFDATA'); 
%         end
%     end
% end

% supersegments feature:
if featparams.segFeature == 1 || featparams.segOnly
    disp('Computing supersegments feature');
    featfile = fullfile(cache_dir, 'feat_SS');
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
       tic;
       if train, CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, 'SS'); end;
       [features, empmean, CRFDATA] = computeSSfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train, featparams.GlobalFeatures);
       e=toc;
       fprintf(' (%0.4f sec)\n', e);
       if numel(cache_dir) && storefeats
           save(featfile, 'features', 'empmean', 'CRFDATA'); 
       end
    end
end
if featparams.segOnly == 1
    empmean = empmean(1);
    return;
end;

% class feature:
if featparams.classFeature == 1
    disp('Computing class (z) feature');
    featfile = fullfile(cache_dir, 'feat_Class');
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
       tic;
       if train, CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, 'Z', gtcols); end;
       [features, empmean, CRFDATA] = computeClassfeature(features, empmean, imlist, CRFDATA, stats, train);
       e=toc;
       fprintf(' (%0.4f sec)\n', e);
       if numel(cache_dir) && storefeats
           save(featfile, 'features', 'empmean', 'CRFDATA'); 
       end
    end
end

% PN feature
if featparams.segFeature == 1
    disp('Computing PN feature');
    featfile = fullfile(cache_dir, 'feat_PN');
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
       tic;
       [features, empmean, CRFDATA] = computePNfeature(features, empmean, imlist, CRFDATA, gamma, ALLPATHS, train);
       e=toc;
       fprintf(' (%0.4f sec)\n', e);
       if numel(cache_dir) && storefeats, save(featfile, 'features', 'empmean', 'CRFDATA'); end;
    end
end

% Coocurrence feature
if featparams.Coocurrence == 1
    disp('Computing Coocourrence feature');
    featfile = fullfile(cache_dir, 'feat_Coocurrence');
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
        tic;
        [features, empmean, CRFDATA] = computeCoocurrencefeature(features, empmean, imlist, CRFDATA, train);
        e=toc;
        fprintf(' (%0.4f sec)\n', e);
        if numel(cache_dir) && storefeats
            save(featfile, 'features', 'empmean', 'CRFDATA'); 
        end;
    end;  
end
  
% Chow-Liu feature
if featparams.chowliuFeature == 1
    disp('Computing Chow-Liu feature');
    featfile = fullfile(cache_dir, 'feat_ChowLiu');
    if exist(featfile, 'file') && numel(cache_dir)
        load(featfile);
    else
       CL = load(fullfile(ALLPATHS.CHOWLIU_PATH,'chowliu_MSRC.mat'));
       tic;
       [features, empmean, CRFDATA] = computeChowLiufeature(features, empmean, imlist, CRFDATA, freq, CL, ALLPATHS, train);
       e=toc;
       fprintf(' (%0.4f sec)\n', e);
       if numel(cache_dir) && storefeats 
           save(featfile, 'features', 'empmean', 'CRFDATA'); 
       end;
    end;  
end;

% detection features:
if numel(ind_classes)
   disp('Computing detection feature');
end;
boxesalldata = [];
if  featparams.detFeature == 1
    for i = 1:length(ind_classes)
        % bbox feature -- bi
        j = ind_classes(i);
        fprintf('  class: %s (%d)\n', classes(j).class, j);
        fprintf('     feature: b\n');
        varname = sprintf('%s-%d', 'bi', j);
        tic;
        [features, empmean, CRFDATA, boxesalldata] = computeBBoxBfeature(features, empmean, imlist, CRFDATA, varname, j, thresholds(j), ALLPATHS, train, boxesalldata);
        % b - z
        if featparams.classFeature == 1
            fprintf('     feature: b-z\n');
            [features, empmean, CRFDATA, boxesalldata] = computeBBoxClassfeature(features, empmean, imlist, CRFDATA, varname, j, ALLPATHS, thresholds(j), train, boxesalldata);
        end
    %     bbox seg -- seg
        fprintf('     feature: seg\n');
        [features, empmean, CRFDATA, boxesalldata] = computeBBoxSPfeature(features, empmean, imlist, CRFDATA, varname, j, thresholds(j), masks, ALLPATHS, train, boxesalldata, featparams.newBBoxSP);
        e=toc;
        fprintf(' (%0.4f sec)\n', e);
    end;
end;

% My code: adding contextual features between bounding boxes
if featparams.SpatialFeatures
    fprintf('Computing contextual Relations:\n');
    boxesalldata = [];
    contextualRelations = {'Above', 'next-to', 'Below', 'overlap'};
%     contextualRelations = {'Above', 'next-to'};
    for j= 1:size(contextualRelations, 2)
        re = char(contextualRelations(j));
        fprintf('     %s\n', re);
        statis = load(fullfile(ALLPATHS.Rstats_PATH,'statistics.mat')); %statistics of relation probs 
        statis = statis.stt;
        [features, empmean, CRFDATA] = coo_bb(features, empmean, imlist, CRFDATA, re, thresholds(j), ALLPATHS, 1, boxesalldata, ind_classes, statis(j));
%         [features, empmean, CRFDATA] = test(features, empmean, imlist, CRFDATA, re, thresholds(j), ALLPATHS, 1, boxesalldata, ind_classes, statis(j));
% %         for i1 = 1:length(ind_classes) - 1
%         for i1 = 1:2
%            ind_class1 = ind_classes(i1);
% %            for i2 = i1:length(ind_classes)
%            for i2 = i1:3
%                ind_class2 = ind_classes(i2);
%                fprintf('     between classes: %d-%d\n', ind_class1, ind_class2);
% %                [features, empmean, CRFDATA] = computeCoocurrenceBBfeaure(features, empmean, imlist, CRFDATA, re, ind_class1, ind_class2, thresholds(j), ALLPATHS, 1, boxesalldata);
%                  [features, empmean, CRFDATA] = test(features, empmean, imlist, CRFDATA, re, ind_class1, ind_class2, thresholds(j), ALLPATHS, 1, boxesalldata);
%            end 
%         end
    end
end
% End of my code

if featparams.SceneFeature
    disp('Computing scene feature');
    potentialfile = fullfile(ALLPATHS.SCENE_PATH, 'potential', 'scenepotentials.mat');
    data = load(potentialfile);
    CRFDATA.NumScenes = length(data.potential);
    tic;
    if train, CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, 'S'); end;
    [features, empmean, CRFDATA] = computeSceneUnaryfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train);
    if featparams.SceneBoostZFeature
       [features, empmean, CRFDATA] = computeSceneZpairwisefeature(features, empmean, imlist, CRFDATA, ALLPATHS, train);
    end;
    if featparams.SceneSupressZFeature
       [features, empmean, CRFDATA] = computeSceneZnegpairwisefeature(features, empmean, imlist, CRFDATA, ALLPATHS, train);
    end;
    e=toc;
    fprintf(' (%0.4f sec)\n', e);
end;
empmean = empmean(1:length(features));

function NumFeatures = getNumFeatures(featparams)

NumFeatures = 0;

if featparams.useSPLocationFeature
    NumFeatures = NumFeatures + 1;
end;
if featparams.useFgBg
   NumFeatures = NumFeatures + 1;   % use for SP and SS
end;
if featparams.useClassSP
   NumFeatures = NumFeatures + 1;   % use for SP and SS
end;
if featparams.segFeature == 1
    NumFeatures = NumFeatures + 5;
    if featparams.useSSLocationFeature
       NumFeatures = NumFeatures + 1;
    end;
else
    NumFeatures = NumFeatures + 3;
end; 

if featparams.chowliuFeature == 1
    NumFeatures = NumFeatures + 1;
end;

NumFeatures = NumFeatures + 3 * length(featparams.ind_classes);

if featparams.SceneFeature == 1
    NumFeatures = NumFeatures + 1;
    if featparams.SceneBoostZFeature == 1
       NumFeatures = NumFeatures + 1;
    end;
    if featparams.SceneSupressZFeature == 1
       NumFeatures = NumFeatures + 1;
    end;    
end;
NumFeatures = NumFeatures + 500;