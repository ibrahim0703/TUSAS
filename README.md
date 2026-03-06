$ python c_main.py
[SYSTEM] VIO Pipeline ba▒lat▒l▒yor (Stereo + IMU + 9D Kalman)...
[IMU] 139600 ▒rnek y▒klendi.
[SYSTEM] 13999 kare i▒lenecek.
[SYSTEM] H▒z tahmini: Kalman filtreli (9D state)
Traceback (most recent call last):
  File "C:\Users\A12540\Desktop\a\TUM\c_main.py", line 245, in <module>
    main()
  File "C:\Users\A12540\Desktop\a\TUM\c_main.py", line 225, in main
    print(f"  [Frame {i:6d}] h▒z={speed:.3f} m/s | "
  File "C:\Program Files\Python310\lib\encodings\cp1254.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 50: character maps to <undefined>

