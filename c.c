#include <stdio.h>

// Hocanın L_6_2 slaytında kullandığı profesyonel erişim makrosu
// M: Matrisin başlangıç adresi (pointer)
// C: Toplam sütun sayısı (ncols)
// I: İstenen satır indeksi (i)
// J: İstenen sütun indeksi (j)
#define MAT_ERISIM(M, C, I, J) (*((M) + (I) * (C) + (J)))

// Fonksiyon sadece düz bir pointer (int *mat) ve boyutları alıyor.
void matrisYazdir(int *mat, int satirSayisi, int sutunSayisi) {
    int i, j;
    printf("--- POINTER ILE MATRIS OKUMA ---\n");
    for (i = 0; i < satirSayisi; i++) {
        for (j = 0; j < sutunSayisi; j++) {
            // Normalde mat[i][j] yazardın. ŞİMDİ YASAK.
            // Makromuzu kullanarak adres matematiği yapıyoruz.
            printf("%d\t", MAT_ERISIM(mat, sutunSayisi, i, j));
        }
        printf("\n");
    }
}

int main() {
    // 2 satır, 3 sütunluk bir matris tanımlıyoruz
    int ordu[2][3] = {
        {10, 20, 30}, // 0. Satır
        {40, 50, 60}  // 1. Satır
    };
    
    // Fonksiyona matrisi gönderirken sadece 0. satır 0. elemanın adresini veriyoruz.
    // L_6_2 slaytındaki kural: &ordu[0][0]
    matrisYazdir(&ordu[0][0], 2, 3);

    return 0;
}
