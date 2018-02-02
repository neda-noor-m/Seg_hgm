function [features, empmean, CRFDATA] = computeSceneUnaryfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train)
% SCENE FEATURE

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname = 'scene';
gamma_score = 1.3;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};    
    data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'potential'); %labelSuperpixel
    U = data.potential;
    NumStates = length(U);
    if featparams.uselogisticScene
       U = 1./(1+exp(-gamma_score*(U)));  % logistic score!
    end;
    
    [vars, ~, blockstart, ~] = setvardata(vars, varname, i, LOC(i) + 1, LOC(i) + 1, NumStates);
    LOC(i) = LOC(i) + 1;

    % Compute Emperical Means
    if train
        label = CRFDATA.labelsS{i};
        empmean(r) = computeEmpiricalMeans(empmean(r), U, label, 1);
    end;
    

    % unary setting
    pntr = blockstart;
    features{r}.sample{i}.local{pntr}.NumStates = NumStates;
    features{r}.sample{i}.local{pntr}.pot = U;

    % pairwise setting
    features{r}.sample{i}.factor= cell(0,0);
end   

CRFDATA = updateData(CRFDATA, vars, LOC, M);
%---------------------------------------  