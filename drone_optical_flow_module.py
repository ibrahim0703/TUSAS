# =============================================================================
# 3. STEREO SGBM VE WLS FİLTRESİ (Endüstri Standardı Derinlik Temizliği)
# =============================================================================
def init_wls_stereo_matcher():
    window_size = 5
    min_disp = 0
    num_disp = 16 * 5 # EuRoC için genelde 80-96 arası iyidir
    
    # Sol Eşleştirici (Ana)
    left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size ** 2,
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY # Daha hassas hesaplama modu
    )
    
    # Sağ Eşleştirici (WLS Filtresi ve Tutarlılık Kontrolü için zorunludur)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    
    # WLS (Weighted Least Squares) Filtresini Kur
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0) # Düzleştirme (Smoothing) katsayısı
    wls_filter.setSigmaColor(1.5) # Kenar koruma hassasiyeti
    
    return left_matcher, right_matcher, wls_filter

# =============================================================================
# 4. ANA DÖNGÜ (Filtrelenmiş Derinlik)
# =============================================================================
def main():
    loader = StereoDataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()

    while True:
        img_left, img_right = loader.get_stereo_pair()
        if img_left is None:
            break
            
        # 1. Sol ve Sağ Disparity'leri Hesapla
        left_disp = left_matcher.compute(img_left, img_right)
        right_disp = right_matcher.compute(img_right, img_left)
        
        # 2. WLS Filtresi ile delikleri yamala ve pürüzsüzleştir
        filtered_disp = wls_filter.filter(left_disp, img_left, None, right_disp)
        
        # 3. Metrik değerleri normale çevir (16'ya böl) ve negatif hataları sıfırla
        filtered_disp_float = filtered_disp.astype(np.float32) / 16.0
        filtered_disp_float[filtered_disp_float < 0] = 0 # Parazitleri kes
        
        # 4. Görselleştirme
        disp_vis = cv.normalize(filtered_disp_float, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        disp_color = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)

        cv.imshow('Sol Kamera (Orijinal)', img_left)
        cv.imshow('WLS Filtreli Kusursuz Derinlik', disp_color)

        if cv.waitKey(30) & 0xff == 27:
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
