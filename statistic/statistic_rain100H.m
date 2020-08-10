
clear all;
close all;

gt_path='/Users/pangbo/ÎÄµµ/Paper/experiment_result/Rain100H/';
gen_path='/Users/pangbo/ÎÄµµ/Paper/experiment_result/SIFT/Rain100H/';

gt_img=dir([gt_path,'norain-*.png']);
gen_img=dir([gen_path,'rain-*.png']);

len=length(gt_img);
total_psnr=0;
total_ssim=0;
fid=fopen('log.txt','w');
for i=1:len
    gt=rgb2ycbcr(im2double(imread([gt_path,gt_img(i).name])));
    gen=rgb2ycbcr(im2double(imread([gen_path,gen_img(i).name])));
    gen=gen(:,:,1);
    gt=gt(:,:,1);
    psnr_val=mean(psnr(gen,gt));
    ssim_val=ssim(gen*255,gt*255);
    fprintf('img=%s,psnr=%.4f,ssim=%.4f \n',gen_img(i).name,psnr_val,ssim_val);
    fprintf(fid,'img=%s,psnr=%.8f,ssim=%.8f \n',gen_img(i).name,psnr_val,ssim_val);
    total_psnr= total_psnr+psnr_val;
    total_ssim= total_ssim+ssim_val;
end
fprintf('avg_psnr:%.4f,avg_ssim:%.4f',total_psnr/len,total_ssim/len)
fprintf(fid,'avg_psnr:%.4f,avg_ssim:%.4f',total_psnr/len,total_ssim/len)