- 普通 sparse transpose conv                                                                                                               
    主要是在已有稀疏支撑上做“升分辨率 + 特征传播”                                                                                            
    更像把已有体素的信息搬到更细网格                                                                                                         
    但如果当前支撑里根本没有某片 blind spot，对那片区域它通常长不出新 occupied voxels                                                        
  - generative sparse transpose conv                                                                                                         
    会显式扩展坐标集合                                                                                                                       
    也就是从一个已有体素，向更细尺度的邻域“提候选体素”                                                                                       
    然后再靠 occupancy head + pruning 决定哪些候选保留                                                                                       
    这才有能力补出 measurement 里没有、但 GT 里有的结构  