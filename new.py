def calculate_3d_points(p0_2d, disparity_map):
    points_3d = []
    valid_points_2d = []
    
    # Derinlik haritasının fiziksel sınırlarını al (480 Yükseklik, 752 Genişlik)
    h, w = disparity_map.shape 
    
    for pt in p0_2d:
        u, v = pt[0][0], pt[0][1] 
        
        # SINIR KONTROLÜ (BOUNDARY CHECK): 
        # Eğer piksel ekranın solundan, sağından, üstünden veya altından dışarı çıktıysa onu anında ÇÖPE AT.
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
            
        d = disparity_map[int(v), int(u)] / 16.0 
        
        # DERİNLİK FİLTRESİ
        if d > 1.0: 
            Z = (f_x * baseline) / d
            if Z > 20.0:
                continue
                
            X = ((u - c_x) * Z) / f_x
            Y = ((v - c_y) * Z) / f_y
            points_3d.append([X, Y, Z])
            valid_points_2d.append([[np.float32(u), np.float32(v)]])
            
    return np.array(points_3d, dtype=np.float32), np.array(valid_points_2d, dtype=np.float32)
