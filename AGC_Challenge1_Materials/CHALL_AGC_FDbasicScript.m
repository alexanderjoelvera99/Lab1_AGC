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
    % Process bbox to have x1 y1 w h
    bbox_v(:, 3) = bbox_v(:, 3) + bbox_v(:, 1);
    bbox_v(:, 4) = bbox_v(:, 4) + bbox_v(:, 2);

    bbox_v_sorted = sortrows(bbox_v, [3, 4], 'descend'); % sorted by size of the faces
    % Take only the 2 biggest faces
    if(num_faces > 2)
        det_faces = bbox_v_sorted(1:2, :);
    else
        det_faces = bbox_v_sorted(:,:);
    end 

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






