# DreamTrainer
Aplikacja webowa do trenowania i generowania obrazów przy użyciu Stable Diffusion.
Działa w Google Colab i pozwala użytkownikom trenować modele na własnych zdjęciach.


Instrukcja:
-Odpalić plik googlecolab.(https://colab.research.google.com/drive/1lkmYIfzGUaUEMUZqhu1x6euU3mmfsyqB?usp=sharing#scrollTo=zh8ZnC_cDPiu) 
-Wgrać resztę plików do środowiska.
-Przeklikać przez wszystkie komórki w pliku
-Odpalić main.py na końcu google colab i wejść w link ngrok


Z przykładowego samplu 20 wygenerowanych obrazów na podstawie 5 zdjęć do nauki twarzy Brada Pitta ręcznie oceniłem czy wygenerowany obraz przypomina Brada Pitta. PROMPT= photo of (concept_name) smiling\n
Stable Diffiusion 1.4: 12/20 obrazów przypomina Orginalne zdjęcia Brada Pitta, 6 przedstawia mężczyzne o podobnych cechach ale nie jest rozpoznawalny brad pitt, 2 zdjęcia nie pokazują nawet mężczyzny\n
Stable Diffiusion 1.3:  Tylko 4/20 obrazów przypominało Brada Pitta 4/20 mężczyznę o podobnych cechach. Aż 12/20 obrazów nie przedstawiało nawet mężczyzny\n
Stable Diffiusion 1.2: 2/20 obrazów przypominało brada pitta, 2/20 innego mężczyzne, 16/20 było zupełnie losowe\n
