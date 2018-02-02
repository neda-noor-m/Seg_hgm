function loss = computeLoss(imlist, CRFDATA, ignore_classes, ind_classes, ALLPATHS)

featparams = CRFDATA.featparams;
featparams = defaultfeatparams(featparams, ind_classes);
GT_PATH = ALLPATHS.GT_PATH; 
% GT_PATH = '/home/neda/Documents/Cources/PHD-thesis/DataBase/MSRC_ObjCategImageDatabase_v2/GroundTruth/';
PIXEL_PATH = ALLPATHS.PIXEL_PATH;
SUPIXEL_PATH = ALLPATHS.SUPIXEL_PATH;
BBOX_PATH = ALLPATHS.BBOX_PATH;
UCM1_PATH = ALLPATHS.UCM1_PATH;
UCM2_PATH = ALLPATHS.UCM2_PATH;
NumStates = CRFDATA.NumClasses;
BinaryStates = 2;
vars = CRFDATA.vars;

[classes, gtcols] = getclassinfo(ALLPATHS.DATASET_NAME);
classes = filterclasses(classes, ignore_classes);   % classes
gtcols(ignore_classes,:) = [];
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

loss.sample = cell(1, length(imlist));
%--------------------------------------------


% Generating the Loss Function
% 11_16,11_17, 11_27, 
% imlist = ['11_16_s'; '11_17_s'; '11_27_s'];

disp('Generating the loss function...')
for i = 1:length(imlist)
    loss.sample{i}.factor = cell(0,0);
     imname = imlist{i};
    
%     if isequal(exist(fullfile(GT_PATH,[imname '_labels.png']),'file'),2) % 2 means it's a file.
    if mod(i-1, 50)==0, fprintf('   image %s (%d/%d)\n', imname, i, length(imlist)); end;
%     GT = imread(fullfile(GT_PATH,[imname '_labels.png']));
    GT = imread(fullfile(GT_PATH,[imname '_GT.bmp']));
    gtim = get_GTclasslabels(GT, gtcols, ignore_classes, ALLPATHS.DATASET_NAME);
    gtim = uint8(gtim);

    if ~featparams.segOnly
       S1 = load(fullfile(UCM1_PATH, [imname '_label.mat']));
       label1 = S1.labels; % superpixel labeling
       labelSUP = CRFDATA.labelsSP{i};
       row1 = CRFDATA.indsSP{i};
    end;
    if featparams.segFeature && ~featparams.spOnly
       S2 = load(fullfile(UCM2_PATH, [imname '_label.mat']));    
       label2 = S2.labels;    
       labelSSEG = CRFDATA.labelsSS{i};
       row2 = CRFDATA.indsSS{i};
    end;
    if numel(ind_classes)
       boxdata = load(fullfile(BBOX_PATH, [imname '.mat']));
       lsvmdata = boxdata.lsvmdata;
    end;
    imsize = size(gtim,1)*size(gtim,2); 

    % SUPERPIXEL LOSS
    [j, blockstart, ~] = getvardata(vars, 'SP', i);
    if j == 0, blockstart = 0; end;
    pntr = blockstart;
        
    if blockstart > 0
        pixelscls = zeros(NumStates, 1);
        for j = 1 : NumStates
            pixelscls(j) = sum(sum(gtim == j));
        end;


        factor = featparams.superpixfactor;   %the weight of SP loss

        for j = 1 : length(labelSUP)
            masksup_j = (label1 == row1(j));  % get the mask for the superpixel
            ind = find(masksup_j > 0);
            indseg = gtim(ind)>0;   % find the part that intersects with non-void groundtruth
            indseg = ind(indseg);
            temp = ones(NumStates,1);
            for k = 1 : NumStates
                cls = k;
                temp(cls) = sum(gtim(indseg) ~= cls) / imsize;
            end;
            temp = temp * factor;
            loss.sample{i}.local{pntr}.NumStates = NumStates;
            loss.sample{i}.local{pntr}.pot = temp;
            pntr = pntr + 1;
        end
    end;
    
    if featparams.spOnly, continue; end;

    % SUPERSEGMENT LOSS
    [j, blockstart, ~] = getvardata(vars, 'SS', i);
    if j == 0, blockstart = 0; end;
    pntr = blockstart;
    if blockstart > 0       
        factor = featparams.supersegfactor;
        for j = 1:length(labelSSEG)
            maskseg_j = (label2 == row2(j));
            ind = find(maskseg_j > 0);
            indseg = gtim(ind)>0;   % find the part that intersects with non-void groundtruth
            indseg = ind(indseg);
            temp = ones(NumStates,1);
            for k = 1 : NumStates
                cls = k;
                temp(cls) = sum(gtim(indseg) ~= cls) / imsize;
            end;
            temp = temp * factor;
            loss.sample{i}.local{pntr}.NumStates = NumStates;
            loss.sample{i}.local{pntr}.pot = temp;
            pntr = pntr + 1;
        end
    end;
    
    if featparams.segOnly, continue; end;

    % Z LOSS
    [j, blockstart, ~] = getvardata(vars, 'Z', i);
    if j == 0, blockstart = 0; end;
    pntr = blockstart;
    if blockstart > 0
        factor = featparams.Zfactor;
        for j = 1 : NumStates
            UNIQ = unique(gtim);
            UNIQ(UNIQ==0) = [];
            temp = zeros(BinaryStates,1);
            if nnz(UNIQ == j) > 0
                temp(1) = 1;
                temp(2) = 0;
            else
                temp(1) = 0;
                temp(2) = 1;
            end
            loss.sample{i}.local{pntr}.NumStates = BinaryStates;
            loss.sample{i}.local{pntr}.pot = factor * temp;
            pntr = pntr + 1;
        end
    end;

    % DETECTION LOSS    
    factor = featparams.Bfactor;
    for j = 1 : length(ind_classes)
        varname = sprintf('%s-%d', 'bi', ind_classes(j));
        [v, blockstart, ~] = getvardata(vars, varname, i);
        if v == 0, blockstart = 0; end;
        pntr = blockstart;
        if blockstart > 0

            cls = geticlass(ALLPATHS.classes, ind_classes(j));
            b = lsvmdata(cls).b;
            boxes = lsvmdata(cls).boxes;
            nboxes = size(b,1);
            if numel(b)
                for k = 1 : nboxes
                    if featparams.pascalloss
                       PascalScore = b(k,1);
                    else
                       PascalScore = 1;
                    end;
                    temp = zeros(BinaryStates,1);
                    temp(1) = PascalScore;
                    temp(2) = 1-PascalScore;
                    loss.sample{i}.local{pntr}.NumStates = BinaryStates;
                    loss.sample{i}.local{pntr}.pot = factor*temp;
                    pntr = pntr + 1;
                end
            end
        end;
    end;

    % Scene loss
    factor = featparams.Scenefactor;
    [sc, blockstart, ~, statestart, stateend] = getvardata(vars, 'scene', i);
    if sc > 0 && factor > 0
        pntr = blockstart;
        NumScenes = stateend - statestart + 1;
        if blockstart > 0
            gtlabel = CRFDATA.labelsS{i};
            temp = ones(NumScenes, 1);
            temp(gtlabel) = 0;
            loss.sample{i}.local{pntr}.NumStates = NumScenes;
            loss.sample{i}.local{pntr}.pot = factor*temp;
        end;
    end;
%  end   
end

fprintf('Loss finished!\n');
