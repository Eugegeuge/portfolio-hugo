import re
import sympy as sp 
import math as mp

def preprocesar_ecuacion(ecuacion):     # 3x debe pasarse como 3*x, con los parentesis igual

    i = 0
    aux = ""
    while(i < len(ecuacion)):
        if (ecuacion[i] in ('x(') and ecuacion[i-1].isdigit()):
            aux += '*'
        aux += ecuacion[i]
        i += 1
    return ''.join(aux)


def cambiar_signos_der (ecuacion):

    changed_signos = ""
    in_parentesis = False
    i = 0

    while(i < len(ecuacion)):

        if (ecuacion[i] == '('):    # Se abre parentesis
            in_parentesis = True
            changed_signos += '('
        
        elif (ecuacion[i] == ')'):    # Se cierra el parentesis
            in_parentesis = False
            changed_signos += ')'
        
        else:
            if (not in_parentesis):
                if (ecuacion[i] == '+'):
                    changed_signos += '-'
                elif (ecuacion[i] == '-'):
                    changed_signos += '+'
                else:
                    changed_signos += ecuacion[i]
            else: 
                changed_signos += ecuacion[i]
        
        i += 1
    return ''.join(changed_signos)


def ecuacion_izq(expresion):

    # Primero sustituimos los espacios (si los hubiera) en blanco de ambos lados (esto no deberia pasar pq daria problemas entre otras cosas)
    expresion = expresion.replace(" ","")

    # Preprocesamos las multiplicaciones de la ecuacion
    expresion = preprocesar_ecuacion(expresion)
    #print(f'Ecuacion con multiplicaciones: {expresion}')

    # Dividimos la ecuacion en dos partes separadas por el '='

    # Si no hay un igual hace la operacion obtenida
    if '=' not in expresion:
        print("OPCION DETECTADA: OPERACION MATEMÁTICA")
        sol = eval(expresion)
        print(f"Solucion = {sol}")

    else:
        print("OPCION DETECTADA: ECUACION")
                                                        
        izq, der = expresion.split('=')                  # Se separa la ecuacion por el = y se trabaja independientemente con ambas

        # Comprobamos si el primer número de la der es positivo o negativo para ponerle el '+' si es positivo
        if der and not der[0] in ('+', '-'):
            der = '+' + der

        # Cambiamos los signos de la derecha ('+' --> '-' y viceversa)
        der = cambiar_signos_der(der)
        #print(f'Derecha cambiada de signo: {der}')

        # Juntamos la ecuacion como si estuviera igualada a 0
        eq = f"{izq}{der}"
        print("Ecuacion inicial igualada a 0:")
        print(f"{eq} = 0")
        sol = res_eq(eq)
        print(f"Solucion = {sol}")

    
def res_eq (ecuacion):
    # se define la variable simbólica
    x = sp.symbols('x')

    # convertir el string en una expresion simbolcia
    ecuacion = sp.sympify(ecuacion)

    # Se resuelve
    sol = sp.solve(ecuacion, x)

    if sol:
        return sol
    else:
        return "NO SOL"

expresion = "69*11=3/x"                # 13x – 5(x + 2) = 4(2x – 1) + 7 No Sol

ecuacion_izq(expresion)

