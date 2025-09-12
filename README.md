# Hotspots_Io
Simulation en Python des points chauds de Io dont le diamètre est inférieur à la résolution du spectro-photomètre étudié. 
L'objectif est d'obtenir, pour différents diamètres de points chauds, le spectre en radiance du point chaud, ainsi que sa température effective. (Température équivalente de rayonnement de corps noir) 

Pour l'instant, les spectres obtenus de jour ressemeblent en forme aux spectres obtenus par l'étude de Clément Royer, mais différent beaucoup en ordre de grandeur.
Ils ont la même structure (pour des longueurs d'onde inférieures à 1 micron, la contribution vient essentiellement du background, c'est à dire de la reflectance du soleil, puis pour des longueurs d'ondes supérieures, c'est le rayonnement thermique du point chaud qui contribue le plus au spectre, jusqu'à éclipser complétement le bruit du spectre de la reflectance. Néanmoins, mes spectres semblent décalés de plusieurs puissances de 10 par rapport aux siens.

Les résulats sont encore moins similaires lorsque le diamètre des hotspots dépasse 10 km. On obtient une radiance excédant 10^3, et en plus, pour des longueurs d'onde inférieures à 1 micron, où la reflectance solaire devrait dominer, ce n'est plus cas, (le bruit est presque totalement éclipsé).

Quant aux spectres de nuit, où le background ne provient plus de la reflectance mais d'un rayonnement de corps noir à 130K, on obtient à nouveau des spectres de forme similaire à ceux de Clément, mais décalés de plusieurs puissances de 10. 

A ce jour, je n'ai toujours pas pu identifier d'où provenait cette différence d'ordre de grandeur.
