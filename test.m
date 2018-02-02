function [features, empmean, CRFDATA, boxesalldata] = test(features, empmean, imlist, CRFDATA, varnameR, ind_class1, ind_class2, threshold, ALLPATHS, train, boxesalldata)
% detection feature: BBOX seg -- SP seg

% ind_class1 = 4;
% ind_class2 = 4;
% varnameR = 'Below';
r = length(features);
BinaryStates = 2;
T_P = zeros(BinaryStates,BinaryStates);
varnameS = 'SP';
varname1 = sprintf('%s-%d', 'bi', ind_class1); 
varname2 = sprintf('%s-%d', 'bi', ind_class2); 

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
% for i = 1:1
    imname = imlist{i};
    if getboxes
       boxdata = load(fullfile(ALLPATHS.BBOX_PATH,[imname '.mat']));
       boxesalldata{i} = boxdata.lsvmdata;
    else
       boxdata.lsvmdata = boxesalldata{i};
    end;
    if strcmp(varnameS, 'SP')
        S = load(fullfile(ALLPATHS.UCM1_PATH,[imname '_label.mat']));
    else
        S = load(fullfile(ALLPATHS.UCM2_PATH,[imname '_label.mat']));
    end;
    SPlabels = S.labels;
    imsize = size(SPlabels);
    lsvmdata = boxdata.lsvmdata;
    
    [~, blockstartB1, ~] = getvardata(vars, varname1, i);
    [~, blockstartB2, ~] = getvardata(vars, varname2, i);
    boxes1 = lsvmdata(ind_class1).boxes;
    nboxes1 = size(boxes1, 1);
    boxes2 = lsvmdata(ind_class2).boxes;
    nboxes2 = size(boxes2, 1);
    b1 = lsvmdata(ind_class1).b;    % b is pascal overlap with groundtruth
    b2 = lsvmdata(ind_class2).b;    
    
    Pair = [];
    for j = 1:nboxes1  % connection matrix [superpixel_id, superedge_id]
        start = 1;
        if ind_class1 == ind_class2
            start = j + 1;
        end
        for k = start:nboxes2
            Pair = [Pair; [j, k]];
        end
    end 
    Pair_Score = zeros(size(Pair, 1),1);
    
    % unary setting
    if nboxes1 ~= 0 && nboxes2 ~= 0
                        pntr = blockstartB1;
                        for k = 1:nboxes1
                            features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
                            features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
                            connFac = [];
                            if numel(Pair)
                               [connFac, ~] = find(Pair(:, 1) == k);
                            end;
                            if isfield(features{r}.sample{i}.local{pntr}, 'connTo')
                               features{r}.sample{i}.local{pntr}.connTo = [features{r}.sample{i}.local{pntr}.connTo; connFac+M(i)]; 
                            else
                               features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
                            end
                            pntr = pntr + 1;
                        end
                        
                        pntr = blockstartB2;
                        for k = 1:nboxes2
                            features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
                            features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
                            connFac = [];
                            if numel(Pair)
                               [connFac, ~] = find(Pair(:, 2) == k);
                            end;
                             if isfield(features{r}.sample{i}.local{pntr}, 'connTo')
                               features{r}.sample{i}.local{pntr}.connTo = [features{r}.sample{i}.local{pntr}.connTo; connFac+M(i)]; 
                            else
                               features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
                            end
                            pntr = pntr + 1;
                        end
                        
                        % pairwise setting
                        for p = 1:size(Pair,1)
                            T_P = zeros(BinaryStates,BinaryStates);
                            T_P (2,2) = 0.9;
                            features{r}.sample{i}.factor{p+M(i)}.pot = reshape(T_P, [BinaryStates^2, 1]);
                            features{r}.sample{i}.factor{p+M(i)}.size = [BinaryStates BinaryStates];    
                        end
    end
    end

CRFDATA = updateData(CRFDATA, vars, LOC, M);

function overlap = computeOverlapping(box1, box2, varnameR)
overlap = 0;
mask = zeros(size(box2));
[row1 , col1] = find (box1 == 1);
[row2 , col2] = find (box2 == 1);  
colF1 = col1(1);
colF2 = col2(1);
colF = 1;
rowF = 1;
res = 1;
switch (varnameR)
    case 'next-to'
        if colF1 <= colF2
            key = 'r';
        else
            key = 'l';
        end
        [colF, colE, res] = computeCol(col1, col2, size(box2, 2), key);
        if res == 1 
            mask(row1(1):row1(end), colF:colE) = 1;
        end
        
    case {'Above', 'Below'}
        if strcmp(varnameR, 'Above')
            key = 'Above';
        else
            key = 'Below';
        end
        [rowF, rowE, res] = computeRow(row1, row2, size(box2, 1), key);
        if res == 1 
            mask(rowF:rowE, col1(1):col1(end)) = 1;
        end
    case 'overlap'
        overlappingMask = box1.*box2;
        overlap = length(find(overlappingMask == 1))/length(find(box2));
end

if res == 1 && ~strcmp(varnameR, 'overlap')
    overlappingMask = mask.*box2;
    overlap = length(find(overlappingMask == 1))/length(find(mask));
    ratio = length(find(box2))/length(find(mask));
    if ratio < 1
       overlap = overlap/ratio;
    end
end
% [r1, c1] = find(mask);
% line([c1(1) c1(1) c1(end) c1(end) c1(1)]', [r1(1) r1(end) r1(end) r1(1) r1(1)]', 'color', 'r', 'linewidth', 2.5);

function [colF, colE, result] = computeCol(col1, col2, endCol, key)
result = 1;    
switch (key)
        case 'l'
            if col1(1) ~= 1
                colF = col1(1) - (col2(end) - col2(1) + 1) ;
                colE = col1(1) - 1;
                if colF < 1
                    colF = 1;
                end
            else
                colF = -1;
                colE = -1;
            end
        case 'r'
            if col1(end) ~= endCol
                colE = col1(end) + (col2(end) - col2(1) + 1);
                colF = col1(end) + 1;
                if colE > endCol
                    colE = endCol;
                end
            else
                colF = -1;
                colE = -1;
            end
end
   if colE < 1 || colF > endCol
        result = -1;
   end
function [rowF, rowE, result] = computeRow(row1, row2, endRow, key)
result = 1;
switch (key)
     case 'Above'
            if row1(1) ~= 1
                rowF = row1(1) - (row2(end) - row2(1) + 1) ;
                rowE = row1(1) - 1;
                if rowF < 1
                   rowF = 1;
                end
            else
                rowF = -1;
                rowE = -1;
            end
        case 'Below'
            if row1(end) ~= endRow
                rowE = row1(end) + (row2(end) - row2(1) + 1);
                rowF = row1(end) + 1;
                if rowE > endRow
                   rowE = endRow;
                end
            else
                rowF = -1;
                rowE = -1;
            end
end
if rowE < 1 || rowF > endRow
    result = -1;
end