Zaprojektuj i zaimplementuj algorytm ewolucyjny. 
Następnie zbadaj jego zbieżność na funkcjach F3 oraz F19 z benchmarku CEC2017 [https://github.com/tilleyd/cec2017-py] dla wymiarowości n = 10. 
Interfejs Twojego algorytmu powinien być taki jak w zadaniu poprzednim. 
Pamiętaj o tym, że prezentowane wyniki powinny być uśrednione.

Ponadto porównaj zbieżność algorytmu ewolucyjnego na wskazanych funkcjach z własną implementacją algorytmu gradientu prostego.


Struktura projektu:

evaluacyjny.py -> implementacja solvera ewolucyjnego

run_tests.py -> wykonanie testów ewolucyjnych

draw_graph.py -> prezentacja wizualna osiagnietych wyników

gradientprosty.py -> implementacja solvera gradientu prostego

gradientprostytests.py -> wykonanie testów gradientem prostym na funkcjach zbliżonych do F3 i F19