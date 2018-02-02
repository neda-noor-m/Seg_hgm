function [features, empmean, CRFDATA] = computeSceneZpairwisefeature(features, empmean, imlist, CRFDATA, ALLPATHS, train)
% Scene pairwise feature: connecting the scene variable and the Zs

r = length(features) + 1;
features{r} = [];
varname1 = 'scene';
varname2 = 'Z';
NumScenes = CRFDATA.NumScenes;
NumClasses = CRFDATA.NumClasses;
BinaryStates = 2;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

T_P = cell(NumScenes, 1);
potentialfile = fullfile(ALLPATHS.SCENE_PATH, 'potential', 'scenepotentials.mat');
potentialfilebin = fullfile(ALLPATHS.SCENE_PATH, 'potential', 'scenepotentialsbinary.mat');
data = load(potentialfile);
pot = data.potential;
data = load(potentialfilebin);
potbin = data.potential;
for i = 1 : NumClasses
    T_P_i = zeros(NumScenes,BinaryStates);
    T_P_i(:, 2) = pot(i, :)';
    if featparams.usesupressinboost
        [ind] = find(potbin(i, :) == 0);
        if numel(ind)
            T_P_i(ind', 2) = -featparams.suppress_gamma;
        end
    end
    T_P{i} = T_P_i;
end

Pair = [(1:NumClasses)', ones(NumScenes, 1)];
if numel(Pair)
    [connFacS, ~] = find(Pair(:, 2) == 1);
end;
Pot = zeros(BinaryStates*NumScenes, size(Pair, 1));
for p = 1:size(Pair,1)
    T_P_p = T_P{Pair(p, 1)}';
    Pot(:, p) = reshape(T_P_p,NumScenes*BinaryStates,1);  
end

for i = 1:length(imlist)
    imname = imlist{i};
    [~, blockstartSc, ~] = getvardata(vars, varname1, i);
    [~, blockstartZ, ~] = getvardata(vars, varname2, i);
    
    Pair = [(1:NumClasses)', ones(NumScenes, 1)];
    
    % Compute Emperical Means
    if train
       label = CRFDATA.labelsS{i};
       Z = CRFDATA.labelsZ{i};
       Z = Z + 1;
       
       temp = 0;
       for j = 1 : size(Z, 1)
           labels = [label, Z(j)];
           temp = computeEmpiricalMeans(temp, T_P{j}, labels, 2);
       end;
       empmean(r) = empmean(r) + temp;
    end;
    
    % unary setting
    
    % Zs
    pntr = blockstartZ;
    for s = 1 : NumClasses
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
        connFac = [];
        if numel(Pair)
           [connFac, ~] = find(Pair(:, 1) == s);
        end;
        features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
        pntr = pntr + 1;
    end;

    % Scenes
    pntr = blockstartSc;
    features{r}.sample{i}.local{pntr}.NumStates = NumScenes;
    features{r}.sample{i}.local{pntr}.pot = zeros(NumScenes, 1);
    features{r}.sample{i}.local{pntr}.connTo = connFacS+M(i);
 
    
    % pairwise setting
    for p = 1:size(Pair,1)
        features{r}.sample{i}.factor{p+M(i)}.pot = Pot(:, p);
        features{r}.sample{i}.factor{p+M(i)}.size = [BinaryStates, NumScenes];    
    end
    
    M(i) = M(i)+size(Pair,1);
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);