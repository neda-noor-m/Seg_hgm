function [features, empmean, CRFDATA, boxesalldata] = computeBBoxSPfeature(features, empmean, imlist, CRFDATA, varnameB, ind_class, threshold, masks, ALLPATHS, train, boxesalldata, newBBoxSP)
% detection feature: BBOX seg -- SP seg
                                            
r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varnameS = 'SP';
gamma_score = 1.5;
betta = 10;
iclass = geticlass(ALLPATHS.classes, ind_class);  
NumStates = CRFDATA.NumClasses;
BinaryStates = 2;
getboxes = 0;
if nargin < 11 || isempty(boxesalldata)
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
    if strcmp(varnameS, 'SP')
        S = load(fullfile(ALLPATHS.UCM1_PATH,[imname '_label.mat']));
    else
        S = load(fullfile(ALLPATHS.UCM2_PATH,[imname '_label.mat']));
    end;
    
    %my code
    I = load(fullfile(ALLPATHS.PIXEL_PATH,[imname '.mat'])); %labelSuperpixel
    P = I.potential;
    U = -P; 
    U = 1./(1+exp(-(U)));  % logistic score!
    %end of my code
    
    SPlabels = S.labels; % which superpixel is each pixel corresponded?
    imsize = size(SPlabels);
    lsvmdata = boxdata.lsvmdata;
    
    if train
        labelSP = CRFDATA.labelsSP{i}; % the lebels of superpixels whose labels are not background
        row1 = CRFDATA.indsSP{i};  % the indexes of superpixels whose labels are not background
        mapping = zeros(size(CRFDATA.labelsSPfull{i}, 1), 1);
        mapping(row1) = (1 : length(row1))';
        [ind] = find(SPlabels);
        SPlabels(ind) = mapping(SPlabels(ind)); % SPlabels shows each pixel correspondes to which superpixel
        labeledPixel = labelSP;
    end;
    
    
    [~, blockstartB, ~] = getvardata(vars, varnameB, i);
    [~, blockstartS, blockendS] = getvardata(vars, varnameS, i);

    Pair_BS = [];  W_PairBS = {};
    boxes = lsvmdata(iclass).boxes; 
    segs = lsvmdata(iclass).segs;
    b = lsvmdata(iclass).b;   % b is pascal overlap with groundtruth
    nboxes = size(boxes, 1);
   
    if nboxes > 0
        % compute segs from masks if missing
        if ~numel(segs) || ~numel(segs{1})
            segim = zeros(imsize(1), imsize(2));
            masks_cls = masks(ind_class);
            for k = 1 : nboxes
                detbox = boxes(k, 1:4);
                detbox = round(detbox);
                comp_id = boxes(k, 5);                
                mask = masks_cls(comp_id).map;
                mask = double(mask > 0.6);
                [segk, ~, ~] = pastebox(segim, detbox, mask);
                segs{k} = segk;
            end;
        end;
        
        %initialize the score of each box
        scores = ones(size(boxes, 1), 1);
    end
    
    for k = 1:nboxes
        box = boxes(k, :);
        box(1:4)=round(box(1:4));
        % clip box to be within image:
        box(1:4) = [max(1, box(1)), max(1, box(2)), min(imsize(2), box(3)), min(imsize(1), box(4))];
        boxmask = zeros(size(SPlabels));
        boxmask(box(2):box(4),box(1):box(3)) = 1; % mask the full bounding box to find intersection with superpixels
        ConLabels = unique(boxmask.*SPlabels);
        ConLabels(ConLabels==0) = [];
        mx = max(segs{k}(:));
        if mx > 0
          segs{k} = segs{k} / mx;
        end;
        segs{k} = full(segs{k});
        for s = 1:length(ConLabels)
            Pair_BS = [Pair_BS;[ConLabels(s), k]]; 
            overlaping = (SPlabels==ConLabels(s)).*boxmask;
            TEMP = segs{k}.*overlaping;
            meta = sum(TEMP(:))/nnz(SPlabels==ConLabels(s)); 
            T_P = zeros(NumStates,BinaryStates);
           
            if featparams.use1_meta == 1            
                T_P(:,BinaryStates) = scores(k)*(1-meta);
            end;
             %my code
%             numSuperpixel = ConLabels(s);
%             if train
%                 numSuperpixel = row1(ConLabels(s));
%             end
%             segScore = 1./(1+exp(-U(numSuperpixel, ind_class)));  % logistic score!
%             T_P(:,1) = ((1-meta) + (1-segScore))/2;
%             T_P(:,BinaryStates) = (meta + segScore)/(2*betta);
%             T_P(ind_class,1) = (meta + segScore)/(2*betta);

            if newBBoxSP
                scoreB = 1./(1+exp(-gamma_score*(boxes(k,end)+threshold)));
                T_P(ind_class,BinaryStates - 1) = scoreB*meta/betta;
                % T_P(ind_class,BinaryStates) = scores(k)*(meta + segScore)/2;
                T_P(ind_class,BinaryStates) = scores(k)*meta*scoreB;
            else
                T_P(ind_class,BinaryStates) = scores(k)*meta;
            end
            %end of my code
            
            % empirical means
            if train
              labels = [labeledPixel(ConLabels(s)), (b(k, 1) >= featparams.boxthresh) + 1];
              empmean(r) = computeEmpiricalMeans(empmean(r), T_P, labels, 2);
            end;
            W_PairBS{size(Pair_BS,1)} = T_P;
        end
    end
    Pair = Pair_BS;

    % Compute Emperical Means

    % unary setting
    pntr = blockstartS;    
    for s = 1:blockendS - blockstartS + 1
        features{r}.sample{i}.local{pntr}.NumStates = NumStates;
        features{r}.sample{i}.local{pntr}.pot = zeros(NumStates, 1);
        connFac = [];
        if numel(Pair)
           [connFac, ~] = find(Pair(:, 1) == s);
        end;
        features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
        pntr = pntr + 1;
    end
    pntr = blockstartB;
    for s = 1:nboxes
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
        connFac = [];
        if numel(Pair)
           [connFac, ~] = find(Pair(:, 2) == s);
        end;
        features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
        pntr = pntr + 1;
    end

    % pairwise setting
    for p = 1:size(Pair,1)
        features{r}.sample{i}.factor{p+M(i)}.pot = W_PairBS{p};
        features{r}.sample{i}.factor{p+M(i)}.size = [NumStates BinaryStates];    
    end

    M(i) = M(i)+size(Pair,1);
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);
