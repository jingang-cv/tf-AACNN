jpegFiles = dir('result_attr_test/*.jpg');
jpegFiles_gt = dir('result_attr_test/gt/*.jpg');
%numfiles = 1984;
numfiles = length(jpegFiles);
ssim_sum = 0;
for k = 1:numfiles 
  path1 = strcat('result_attr_test/', jpegFiles(k).name);
  path2 = strcat('result_attr_test/gt/', jpegFiles_gt(k).name);

  im1 = imread(path1);
  im2 = imread(path2);
  ssim_sum = ssim(im1, im2) + ssim_sum;
end
ssim_sum = ssim_sum / numfiles 
g = sprintf('%d ', ssim_sum);
fprintf('SSIM: %s\n', g)
