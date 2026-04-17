#include <stdio.h>

int main() {
    // 1. GÜVENLİ BÖLGE (Stack Array)
    char degisebilir_kelime[] = "ZAFER";
    
    // 2. MAYINLI BÖLGE (Read-Only Pointer)
    char *beton_kelime = "ZAFER";

    printf("--- OPERASYON BASLIYOR ---\n");

    // Stack'teki kelimeyi değiştiriyoruz. Sorun çıkmayacak.
    degisebilir_kelime[0] = 'B';
    printf("1. Kelime degisti: %s\n", degisebilir_kelime);

    // Şimdi Read-Only bölgedeki kelimeyi değiştirmeye çalışıyoruz.
    printf("2. Kelime degistiriliyor... (Burada cokecek)\n");
    
    // Tetiği Çek:
    beton_kelime[0] = 'B'; 

    // Program çöktüğü için bu satır asla ekrana basılamayacak:
    printf("Eger bunu okuyorsan basardin: %s\n", beton_kelime);

    return 0;
}
