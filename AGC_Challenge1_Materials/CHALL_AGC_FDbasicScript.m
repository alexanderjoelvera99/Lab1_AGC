 % Basic script for Face Detection Challenge
% --------------------------------------------------------------------
% AGC Challenge  
% Universitat Pompeu Fabra
% Authors
% Juan Marquerie (192631)
% Xavier Gallardo (205680)
% Alexander Vera (206074)

% Load challenge Training data
load AGC_Challenge1_Training

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge\images\'
imgPath = [];

% Initialize results structure
DetectionSTR = struct();

% Initialize timer accumulator
total_time = 0;
faceDetector = vision.CascadeObjectDetector("FrontalFaceCART");
faceDetector.MinSize = [98 100];
faceDetector.MaxSize = [3500 4000];
faceDetector.MergeThreshold = 15;
faceDetector.ScaleFactor = 1.05;

% Process all images in the Training set
for j = 1 : length( AGC_Challenge1_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC_Challenge1_TRAINING(j).imageName ));    
    j
    % try
    % Timer on
    tic;
    
    % ###############################################################
    % Your face detection function goes here. It must accept a single
    % input parameter (the input image A) and it must return one or
    % more bounding boxes corresponding to the facial images found 
    % in image A, specificed as [x1 y1 x2 y2]
    % Each bounding box that is detected will be indicated in a 
    % separate row in det_faces
    
    %det_faces = MyFaceDetectionFunction( A );        
    % ###############################################################
    bbox_v = faceDetector(A);
    num_faces = size(bbox_v);

    if num_faces(1) == 0
        det_faces = bbox_v;
        continue;
    end

    det_faces = zeros(num_faces);
    % Al final d'aquest for loop, la variable det_faces tindrà les
    % coordenades de cadascuna de les boxes que hagi detectat com a cares
    % (una per fila)

    for i = 1:num_faces(1)
        tmp = bbox_v(i,:);
        tmp(3) = tmp(3) + tmp(1);
        tmp(4) = tmp(4) + tmp(2);
        det_faces(i, :) =  tmp;

    end

    %Ordenar el det_faces de box més gran a més petita, i tallar-la a les
    %dues primeres files

    sizes = zeros(num_faces(1,1));
    
    for i = 1:size(sizes)
        sizes(i) = bbox_v(i,3) * bbox_v(i,4);
    end 
    
    max_pos = 0;
    secondmax = 0;
    for i=1:size(sizes)
        if sizes(i)>max_pos
            max_pos = i;
        end
    end

    for i=1:size(sizes)
        if sizes(i)>secondmax && sizes(i) ~= sizes(max_pos)
            secondmax = i;
        end
    end
    
    det_faces_temp = zeros(2,4);
    if secondmax ~= 0
        %agafo les files de det_faces amb index max_pos i secondmax
        det_faces_temp(1,:) = det_faces(max_pos,:);
        det_faces_temp(2,:) = det_faces(secondmax,:);
    else
        %agafo només les files de det_faces amb index max_pos
        det_faces_temp = det_faces(max_pos,:);
    end

    det_faces = det_faces_temp;

    % Update total time
    tt = toc;
    total_time = total_time + tt;
        
    %catch
        % If the face detection function fails, it will be assumed that no
        % face was detected for this input image
     %   det_faces = [];
    %end

    % Store the detection(s) in the resulst structure
    DetectionSTR(j).det_faces = det_faces;
end
   
% Compute detection score
FD_score = CHALL_AGC_ComputeDetScores(...
    DetectionSTR, AGC_Challenge1_TRAINING, 0);

% Display summary of results
fprintf(1, '\nF1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FD_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );






