function plot_2d_ROI(vol,mask,outputfolder,w_s,tp,onset)

   % MRI_PATCH
     im_path1=[outputfolder,'\',tp,'_ROI_lesion.png'];
    % mask ROI
    im_path5=[outputfolder,'\',tp,'_lesion_mask.png'];
    % overlay
    im_path6=[outputfolder,'\',tp,'_overlay.png'];
        
   
    [bb_vol,bb_mask] = boundingbox2(vol,mask,w_s,'cropz','off');
    bb_vol=bb_vol(:,:,round(size(bb_vol,3)/2)+onset);
    bb_mask=bb_mask(:,:,round(size(bb_mask,3)/2)+onset);
    
        
    imagesc(bb_vol'); colormap(gray);colorbar;axis off 
    set(gcf, 'Windowstyle', 'docked','Visible', 'off')
    saveas(gca,im_path1);
    close all;
   
    % mask ORI
    imagesc(bb_mask'); colormap(jet);colorbar;axis off 
    set(gcf, 'Windowstyle', 'docked','Visible', 'off')
    saveas(gca,im_path5);
    close all;
    
    
    %Super poner mask ROI and collage
    A=imread(im_path1);
    imshow(A, 'InitialMag', 'fit');hold on
    set(gcf, 'Windowstyle', 'docked','Visible', 'off')
    B=imread(im_path5);b=imshow(B, 'InitialMag', 'fit') ;
    set(b, 'AlphaData', B(:,:,1)>0);
    width=900;height=900;x0=70;y0=100;
    set(gcf,'position',[x0,y0,width,height])
    saveas(gca,im_path6)
    delete(im_path5)
   close all
    

end 