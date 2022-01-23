function bbox = MyFaceDetectionFunction(A)
    % Create a cascade detector object.
    faceDetector = vision.CascadeObjectDetector();
    faceDetector.MinSize = [60 60];
    faceDetector.MergeThreshold = 10;
    faceDetector.ScaleFactor = 1.1;
    
    bbox_v = faceDetector(A);

    num_faces = size(bbox_v)

    if num_faces(1) == 0
        bbox = bbox_v;
        return
    end

    bbox = zeros(num_faces);

    for i = 1:num_faces(1)
        tmp = bbox_v(i,:);
        tmp(3) = tmp(3) + tmp(1);
        tmp(4) = tmp(4) + tmp(2);
        bbox(i, :) =  tmp;

    end
    
end