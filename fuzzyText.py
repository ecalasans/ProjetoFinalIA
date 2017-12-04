# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl


def fuzzyRelText(relTotal=0, relRadical=0, relNoMatch=0):
    # Cria as variáveis fuzzy:  Antecedentes e Consequente
    # Antecedentes
    total = ctrl.Antecedent(np.arange(start=0, stop=1.1, step=0.1), 'Total')
    radical = ctrl.Antecedent(np.arange(start=0, stop=1.1, step=0.1), 'Radical')
    noMatch = ctrl.Antecedent(np.arange(start=0, stop=1.1, step=0.1), 'NoMatch')

    #Consequentes
    nivelRelevancia = ctrl.Consequent(np.arange(start=0.0, stop=1.1, step=0.1), 'Relevancia')


    #Funções
    #Total
    total['nRelevante'] = fuzzy.trimf(total.universe, [0, 0, 0.8])
    total['relevante'] = fuzzy.trimf(total.universe, [0.2, 1, 1])

    #Radical
    radical['nRelevante'] = fuzzy.trimf(radical.universe, [0, 0, 0.8])
    radical['relevante'] = fuzzy.trimf(radical.universe, [0.2, 1, 1])

    #NoMatch
    noMatch['nRelevante'] = fuzzy.trimf(noMatch.universe, [0, 0, 0.8])
    noMatch['relevante'] = fuzzy.trimf(noMatch.universe, [0.2, 1, 1])

    #resultado
    nivelRelevancia['poucoRelevante'] = fuzzy.trimf(nivelRelevancia.universe, [0, 0, 0.4])
    nivelRelevancia['relevante'] = fuzzy.trimf(nivelRelevancia.universe, [0.1, 0.5, 0.9])
    nivelRelevancia['muitoRelevante'] = fuzzy.trimf(nivelRelevancia.universe, [0.6, 1, 1])

    # Regras
    r1 = ctrl.Rule(antecedent=total['relevante'] & radical['relevante'] & noMatch['nRelevante'],
                   consequent=nivelRelevancia['muitoRelevante'])
    r2 = ctrl.Rule(antecedent=total['nRelevante'] & radical['nRelevante'] & noMatch['relevante'],
                   consequent=nivelRelevancia['poucoRelevante'])
    r3 = ctrl.Rule(antecedent=total['relevante'] & radical['relevante'] & noMatch['nRelevante'],
                   consequent=nivelRelevancia['relevante']%0.8)
    r4 = ctrl.Rule(antecedent=total['relevante'] & radical['relevante'] & noMatch['relevante'],
                   consequent=nivelRelevancia['relevante']%0.6)
    r5 = ctrl.Rule(antecedent=total['relevante'] & radical['nRelevante'] & noMatch['nRelevante'],
                   consequent=nivelRelevancia['relevante']%0.7)
    r6 = ctrl.Rule(antecedent=total['nRelevante'] & radical['relevante'] & noMatch['nRelevante'],
                   consequent=nivelRelevancia['relevante']%0.5)
    r7 = ctrl.Rule(antecedent=total['nRelevante'] & radical['nRelevante'] & noMatch['nRelevante'],
                   consequent=nivelRelevancia['poucoRelevante'])
    
    

    #Cria máquina de inferência
    controleRelevancia = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7])

    #Computa a inferência
    resultado = ctrl.ControlSystemSimulation(control_system=controleRelevancia)

    #Entrada de dados
    resultado.input['Total'] = relTotal
    resultado.input['Radical'] = relRadical
    resultado.input['NoMatch'] = relNoMatch

    #Defuzzificação
    resultado.compute()

    return resultado.output['Relevancia']

