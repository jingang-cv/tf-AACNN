jpegFiles = dir('result_attr_test/*.jpg');
jpegFiles_gt = dir('result_attr_test/gt/*.jpg');
numfiles = length(jpegFiles);
psnr = 0;
for k = 1:numfiles 
  path1 = strcat('result_attr_test/', jpegFiles(k).name);
  path2 = strcat('result_attr_test/gt/', jpegFiles_gt(k).name);

  im1 = imread(path1);
  im2 = imread(path2);
  psnr = compute_psnr(im1, im2) + psnr;
end
psnr = psnr / numfiles 
g = sprintf('%d ', psnr);
fprintf('PSNR: %s\n', g)
