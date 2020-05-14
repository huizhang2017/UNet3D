import pandas as pd

if __name__ == '__main__':
    
    image_path = '/data/data/Beijing_CBCT_Unilateral_Cleft_Lip_and_Palate/GroundTruth/flip_Res0p4_smoothed/'
    
    image_filename = []
    annotation_filename = []
    
    for i in range(1, 31):
        image_filename.append('NORMAL0{0}_cbq-n3-7.hdr'.format(i))
        annotation_filename.append('NORMAL0{0}-ls-corrected-ordered-smoothed.nrrd'.format(i))
        
    df = pd.DataFrame(list(zip(image_filename, annotation_filename)), columns=['image', 'label'])
    df['image'] = image_path+ df['image']
    df['label'] = image_path+ df['label']
    df.to_csv('data_list.csv', header=True, index=False)
        
        
        
