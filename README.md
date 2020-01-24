Para ejecutar el predictor, debe tener instalado el compilador de python en su equipo.

El script que debe correr es main.py, de la siguiente forma:

    python main.py N Np maxIter C

Donde
N: numero de samples para entrenar la red
	Np: cantidad de particulas
	maxIter: número de iteraciones para qpso
	C: parámetro de penalidad de la pseudoinversa

Ejemplo

    python main.py 3000 20 30 10000


Una vez ejecutado el script, al final le pedira que ingrese datos para predecir.
Aquí puede ingresar una tupla de datos para predecir de la siguiente forma:

	282,tcp,ftp,SF,164,601,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,1,3,1,0.00,0.00,0.00,0.00,0.33,0.67,0.00,255,1,0.00,0.04,0.00,0.00,0.00,0.00,0.03,0.00