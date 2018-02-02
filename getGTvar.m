function CRFDATA = getGTvar(CRFDATA, imlist, ALLPATHS, varname, gtcols)

if nargin < 5
   gtcols = [];
end;

if ~isfield(CRFDATA, 'labelsSP')
   CRFDATA.labelsSP = cell(length(imlist), 1);
   CRFDATA.labelsSPfull = cell(length(imlist), 1);
   CRFDATA.indsSP = cell(length(imlist), 1);
end;
if ~isfield(CRFDATA, 'labelsSS')
   CRFDATA.labelsSS = cell(length(imlist), 1);
   CRFDATA.labelsSSfull = cell(length(imlist), 1);
   CRFDATA.indsSS = cell(length(imlist), 1);
end;
if ~isfield(CRFDATA, 'labelsZ')
   CRFDATA.labelsZ = cell(length(imlist), 1);
end;
if ~isfield(CRFDATA, 'labelsS')
   CRFDATA.labelsS = cell(length(imlist), 1);
end;

[CRFDATA, loaded] = loadFile(ALLPATHS, varname, CRFDATA);
if loaded, return; end;

for i = 1:length(imlist)
    imname = imlist{i};    
    
    switch varname
       case 'SP'
          SP = load(fullfile(ALLPATHS.PIXEL_PATH,[imname '_labels.mat']), 'labelSeg');
          labelSP = SP.labelSeg;  % each component denotes the GT class for that superpixel
          [row, ~, ~] = find(labelSP);   % find superpixels with class non-zero (ignore 'void')
          CRFDATA.labelsSPfull{i} = labelSP;
          labelSP = labelSP(row,:);
	      CRFDATA.labelsSP{i} = labelSP;
          CRFDATA.indsSP{i} = row;
       case 'SS'
          SS = load(fullfile(ALLPATHS.SUPIXEL_PATH,[imname '_labels.mat']), 'labelSeg');
          labelSS = SS.labelSeg;  % each component denotes the GT class for that superpixel
          [row, ~, ~] = find(labelSS);   % find superpixels with class non-zero (ignore 'void')
	      CRFDATA.labelsSSfull{i} = labelSS;
          labelSS = labelSS(row,:);
	      CRFDATA.labelsSS{i} = labelSS;
	      CRFDATA.indsSS{i} = row;
       case 'Z'
          gtfile = fullfile(ALLPATHS.GT_PATH,[imname '_GT.bmp']);
          labelsZ = getZfromGT(gtfile, gtcols);  % Z(i)=0 if class not present and 1 otherwise
          CRFDATA.labelsZ{i} = labelsZ;
	case 'S'
	      data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'gtlabel');
          CRFDATA.labelsS{i} = data.gtlabel;
    end;
    
end   

saveFile(ALLPATHS, varname, CRFDATA)

function [CRFDATA, loaded] = loadFile(ALLPATHS, varname, CRFDATA)

loaded = 0;
    switch varname
       case 'SP'
          fileName = fullfile(ALLPATHS.PIXEL_PATH, 'all_labels.mat');
          if ~exist(fileName, 'file'), return; end;
          data = load(fileName);
          CRFDATA.labelsSPfull = data.labelsSPfull;
          CRFDATA.labelsSP = data.labelsSP;
          CRFDATA.indsSP = data.indsSP;
       case 'SS'
          fileName = fullfile(ALLPATHS.SUPIXEL_PATH, 'all_labels.mat');
          if ~exist(fileName, 'file'), return; end;
          data = load(fileName);
          CRFDATA.labelsSSfull = data.labelsSSfull;
          CRFDATA.labelsSS = data.labelsSS;
          CRFDATA.indsSS = data.indsSS;
       case 'Z'
          fileName = fullfile(ALLPATHS.GT_PATH,'all_labels.mat');
          if ~exist(fileName, 'file'), return; end;
          data = load(fileName);
          CRFDATA.labelsZ = data.labelsZ;
	case 'S'
          fileName = fullfile(ALLPATHS.SCENE_PATH,'all_labels.mat');
          if ~exist(fileName, 'file'), return; end;
          data = load(fileName);
          CRFDATA.labelsS = data.labelsS;
        otherwise
            error('data doesnt exist')
    end;
    
    loaded = 1;
    
    
function saveFile(ALLPATHS, varname, CRFDATA)

    switch varname
       case 'SP'
          fileName = fullfile(ALLPATHS.PIXEL_PATH, 'all_labels.mat');
          labelsSPfull = CRFDATA.labelsSPfull;
          labelsSP = CRFDATA.labelsSP;
          indsSP = CRFDATA.indsSP;
          save(fileName, 'labelsSPfull', 'labelsSP', 'indsSP')
       case 'SS'
          fileName = fullfile(ALLPATHS.SUPIXEL_PATH, 'all_labels.mat');
          labelsSSfull = CRFDATA.labelsSSfull;
          labelsSS = CRFDATA.labelsSS;
          indsSS = CRFDATA.indsSS;
          save(fileName, 'labelsSSfull', 'labelsSS', 'indsSS')
       case 'Z'
          fileName = fullfile(ALLPATHS.GT_PATH, 'all_labels.mat');
          labelsZ = CRFDATA.labelsZ;
          save(fileName, 'labelsZ')
	case 'S'
          fileName = fullfile(ALLPATHS.SCENE_PATH,'all_labels.mat');
          labelsS = CRFDATA.labelsS;
          save(fileName, 'labelsS')
        otherwise
            error('data doesnt exist')
    end;
    