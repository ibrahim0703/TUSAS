#ifndef STACK_H
#define STACK_H

#include <iostream>
#include <vector>

template <typename T>
class Stack {
private:
    // Stack elemanlarını tutmak için en basit ve güvenli yöntem olarak vector kullanıyoruz.
    std::vector<T> elements;

public:
    // Eleman ekleme
    void push(const T& item) {
        elements.push_back(item);
    }

    // Üstten eleman çıkarma (boş değilse)
    void pop() {
        if (!elements.empty()) {
            elements.pop_back();
        }
    }

    // En üstteki elemanı döndürme
    T top() {
        return elements.back();
    }

    // Stack boş mu kontrolü
    bool empty() {
        return elements.empty();
    }

    // Stack boyutu
    size_t size() {
        return elements.size();
    }

    // İstenilen formatta yazdırma fonksiyonu
    void display() {
        std::cout << "Stack Elements: ";
        for (size_t i = 0; i < elements.size(); ++i) {
            std::cout << elements[i] << " ";
        }
        std::cout << std::endl;
    }

    // İki stack'i birleştiren friend fonksiyonu (p ve q)
    // Template sınıflarda friend fonksiyonu doğrudan sınıf içinde tanımlamak 
    // derleme hatalarını önlemek için en pratik yoldur.
    friend Stack<T> operator+(const Stack<T>& p, const Stack<T>& q) {
        Stack<T> result;
        
        // Önce ilk stack'in (p) elemanlarını ekle
        for (size_t i = 0; i < p.elements.size(); ++i) {
            result.push(p.elements[i]);
        }
        
        // Sonra ikinci stack'in (q) elemanlarını ekle
        for (size_t i = 0; i < q.elements.size(); ++i) {
            result.push(q.elements[i]);
        }
        
        return result;
    }
};

#endif