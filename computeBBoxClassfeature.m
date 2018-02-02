function [features, empmean, CRFDATA, boxesalldata] = computeBBoxClassfeature(features, empmean, imlist, CRFDATA, varnamebox, ind_class, ALLPATHS, threshold, train, boxesalldata)
% detection feature: bi--z

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
BinaryStates = 2;
T_P = zeros(BinaryStates,BinaryStates);
T_P(1,2) = -1000;
varnameZ = 'Z';
getboxes = 0;
if nargin < 10 || isempty(boxesalldata)
   boxesalldata = cell(length(imlist), 1);
   getboxes = 1;
end;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};
    if getboxes
       boxdata = load(fullfile(ALLPATHS.BBOX_PATH,[imname '.mat']));
       boxesalldata{i} = boxdata.lsvmdata;
    else
       boxdata.lsvmdata = boxesalldata{i};
    end;
    
    if train        
        Z = CRFDATA.labelsZ{i} + 1;
    end;
    
    [~, blockstartB, ~] = getvardata(vars, varnamebox, i);
    [~, blockstartZ, ~] = getvardata(vars, varnameZ, i);

    iclass = geticlass(ALLPATHS.classes, ind_class);
    lsvmdata = boxdata.lsvmdata;
    boxes = lsvmdata(iclass).boxes;
    b = lsvmdata(iclass).b;    % b is pascal overlap with groundtruth
    nboxes = size(boxes, 1);
    
    Pair = [];
    if nboxes
       Pair = [ind_class*ones(nboxes, 1), (1:nboxes)'];
    end;

    % Compute Emperical Means (fixed)
    if train
        for k = 1:nboxes
	     labels = [Z(ind_class), (b(k, 1) >= featparams.boxthresh) + 1];
	     empmean(r) = computeEmpiricalMeans(empmean(r), T_P, labels, 2);
        end
    end;

    % unary setting
    s = blockstartZ + ind_class - 1;
    features{r}.sample{i}.local{s}.NumStates = BinaryStates;
    features{r}.sample{i}.local{s}.pot = zeros(BinaryStates, 1);
    connFac = [];
    if numel(Pair)
       [connFac, ~] = find(Pair(:, 1) == ind_class);
    end;
    features{r}.sample{i}.local{s}.connTo = connFac+M(i);

    pntr = blockstartB;
    for k = 1:nboxes
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
        connFac = [];
        if numel(Pair)
           [connFac, ~] = find(Pair(:, 2) == k);
        end;
        features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
        pntr = pntr + 1;
    end


    % pairwise setting
    for p = 1:size(Pair,1)
        features{r}.sample{i}.factor{p+M(i)}.pot = reshape(T_P, [BinaryStates^2, 1]);
        features{r}.sample{i}.factor{p+M(i)}.size = [BinaryStates BinaryStates];    
    end
    
    M(i) = M(i)+size(Pair,1);
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);
