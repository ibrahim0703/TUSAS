#include <stdio.h>

// Geçen derste öğrendiğimiz o ölümcül matris erişim makrosu
#define MAT(M, C, I, J) (*((M) + (I) * (C) + (J)))

// Fonksiyon sadece adresi (int *mat) ve boyutu (N) alıyor
void sifirBombardimani(int *mat, int N) {
    int i, j;
    int hedef_satir = -1; // Henüz bulamadık
    int hedef_sutun = -1; // Henüz bulamadık

    // 1. FAZ: İSTİHBARAT (Sadece bak, dokunma)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (MAT(mat, N, i, j) == 0) {
                hedef_satir = i;
                hedef_sutun = j;
                // "1 yerde 0 olacak" dediği için hedefi bulduk, 
                // daha fazla aramaya gerek yok. (Optimizasyon)
                break; 
            }
        }
    }

    // Eğer matriste hiç 0 yoksa (hedef hala -1 ise) operasyonu iptal et
    if (hedef_satir == -1) return;

    // 2. FAZ: OPERASYON (Yok et)
    // Sadece hedef satırdaki tüm sütunları sıfırla
    for (j = 0; j < N; j++) {
        MAT(mat, N, hedef_satir, j) = 0;
    }
    
    // Sadece hedef sütundaki tüm satırları sıfırla
    for (i = 0; i < N; i++) {
        MAT(mat, N, i, hedef_sutun) = 0;
    }
}

// Matrisi ekrana basmak için yardımcı fonksiyon
void matrisYazdir(int *mat, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d\t", MAT(mat, N, i, j));
        }
        printf("\n");
    }
    printf("------------------\n");
}

int main() {
    int N = 3; // 3x3 bir matris
    
    // İçinde sadece 1 tane sıfır olan test matrisi
    int ordu[3][3] = {
        {1, 2, 3},
        {4, 0, 6}, // 0 burada, merkezde (Satır 1, Sütun 1)
        {7, 8, 9}
    };

    printf("--- OPERASYON ONCESI ---\n");
    matrisYazdir(&ordu[0][0], N);

    // Fonksiyona 0. elemanın adresini fırlatıyoruz
    sifirBombardimani(&ordu[0][0], N);

    printf("--- OPERASYON SONRASI ---\n");
    matrisYazdir(&ordu[0][0], N);

    return 0;
}
