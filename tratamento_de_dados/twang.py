# %%
import pandas as pd 
import random as rand
import numpy as np

# %%
# função que calcula o "Gini_pai" em cada nó
# o gini pai é: 1-((n_neg/ntotal)^2 + (n_pos/ntotal)^2)

def ginipai(set, var):

    classes = set[var].value_counts(normalize = True)
    gini_pai = 1- (classes**2).sum()
    
    return(gini_pai)

# %%
# função de cálculo do gini_ponderado - combinação dos ginis_filhos 

def ginino(set, var, limiar, var_alvo, subvar_pos):

    esquerda = set[set[var] < limiar][var_alvo]
    direita = set[set[var] >= limiar][var_alvo]
    
    #proteção contra grupos vazios 
    if len(esquerda) == 0 or len(direita) == 0: 
        return np.inf
    
    # p0_es: amostras_abaixo_do_limiar_que_pertencem_a_classe_0 / total_de_amostras_abaixo_do_limiar
    # p1_es: amostras_abaixo_do_limiar_que_pertencem_a_classe_1 / total_de_amostras_abaixo_do_limiar
    p0_es = (esquerda != subvar_pos).astype(float).sum()/len(esquerda)
    p1_es = (esquerda == subvar_pos).astype(float).sum()/len(esquerda)
    gini_es = 1 - (p0_es**2 + p1_es**2)

    # p0_di: amostras_acima_do_limiar_que_pertencem_a_classe_0 / total_de_amostras_acima_do_limiar
    # p1_di: amostras_acima_do_limiar_que_pertencem_a_classe_1 / total_de_amostras_acima_do_limiar
    p0_di = (direita != subvar_pos).astype(float).sum()/len(direita)
    p1_di = (direita == subvar_pos).astype(float).sum()/len(direita)
    gini_di = 1 - (p0_di**2 + p1_di**2)
    
    
    
    #cálculo do gini ponderado 
    gini_pond = gini_es*(len(esquerda)/len(set)) + gini_di*(len(direita)/len(set))

    return(gini_pond)

# %%
# Função usada para realizar o filtro de decisão em uma árvore do modelo random forest 
# MODELADO PARA FILTROS BINOMIAIS, 0 OU 1, NÃO FUNCIONA PARA MAIS TIPAGENS

def treeFunc(set, var_alvo, subvar_pos):
    # cálculo do gini para a proporção de amostras positivas e negativas no nó 
    gini_pai = ginipai(set, var_alvo)
    # primeio critério de "florerimento" 
    if gini_pai ==0:
        leaf = {"type": "leaf", "split": [len(set[set[var_alvo] != subvar_pos]), len(set[set[var_alvo] == subvar_pos])], "gini": gini_pai}
        return leaf 
    # segundo critério de florecimento: mínimo de 1 amostra por folha, seguindo o modelo dos autores 
    if len(set) <= 1:
        leaf = {"type": "leaf", "split": [len(set[set[var_alvo] != subvar_pos]), len(set[set[var_alvo] == subvar_pos])], "gini": gini_pai}
        return leaf 
    

    # sorteio das variáveis analisadas no para o split 
    # a quantidade de variáveis sorteadas é n_vars^0.5
    vars = set.columns.tolist()
    vars.remove(var_alvo)
    nsort = int(((len(vars))**0.5)//1)
    vars_sort = rand.sample(vars, nsort)
    

    # definição da variável ótima "feature", limiar "threshold" e do gini ponderadado dos ginis_filhos "gini_no"
    gini_no = gini_pai
    var_opt = None
    x_opt = None

    # looping usado para a avaliação de todos os possíveis limiares para cada variável sorteada
    # Foi adicionada uma limitação de no máximo 100 limiares por variável com a intenção de diminuir o tempo de execução da floresta 
    for i in vars_sort:
        valores = np.sort(set[i].unique())
        if len(valores)> 100:
            valores = np.linspace(min(valores), max(valores), 100)
        limiares = [(valores[v] + valores[v+1])/2 for v in range(0,len(valores)-1)]
        for j in limiares:
            result = ginino(set, i, j, var_alvo, subvar_pos)
            if result < gini_no:
                gini_no = result
                var_opt = i
                x_opt = j
    
    # terceiro critério de "florecimento": melhor gini_ponderado > gini_nó 
    # no caso o looping não conseguiu encontrar um candidato válido para separação 
    if gini_no == gini_pai:
        leaf = leaf = {"type": "leaf", "split": [len(set[set[var_alvo] != subvar_pos]), len(set[set[var_alvo] == subvar_pos])], "gini": gini_pai}
        return leaf
    
    # separação das amostras em esquerda e direita de acordo com o melhor limiar da melhor variável sorteada 
    left = set[set[var_opt] < x_opt]
    right = set[set[var_opt] >= x_opt]

    # linha de código EXTREMAMENTE IMPORTANTE, é aqui que a função se torna autorecursiva, recorrendo a si mesma dentro da execução 
    # o grupo de variáveis da esqueda irá passar pelo mesmo processo, assim como o grupo da direita
    # o processo só será interrompido caso a função encontre um dos critérios de parada definidos 
    left_child = treeFunc(left, var_alvo, subvar_pos)
    right_child = treeFunc(right, var_alvo, subvar_pos)
    
    # node com os parâmetros de interesse:
    # - "type": defini se o dicionário é nó ou folha, sempre recrutado nas funções de avaliação da árvore 
    # - "left_child": ramo esquerdo, contem outros divérssos nós e folhas dentro de si mesmo 
    # - "right_child": análogo ao ramo esquerdo
    # - "Gini": o gini ponderado dos filhos, gini do split
    # - "Gini_parent": gini pai, calculado com a proporção de amostras negativas e positivas 
    # - "split": quantidade de variáveis enviadas para a esquerda do ramo e quantidade de variáveis enviadas para a direita do ramo
    # - "feature": variável usada na divisão 
    # - "threshold": limiar usado na divisão 
    node = {"type": "node", "left_child": left_child, "right_child": right_child, 
            "Gini": gini_no, "Gini_parent": gini_pai, "split": [len(left), len(right)] ,
              "feature": var_opt, "threshold": x_opt}
            

    # retorno da função: um dicionário de dicionários 
    return node

# %%
# Cálculo da árvore mais profunda 
# função pra calcular a profundidade máxima de uma árvore
def deepTree(tree):

    if tree["type"] == "leaf":
        return 1
    
    
    right = 1 + deepTree(tree["left_child"])
    left = 1 + deepTree(tree["right_child"])

    return max(left, right)


# %%
# calculo da quantidade de nós + quantidade de folhas 
def totalNodes(tree):

    if tree["type"] == "leaf":
        return 0
    
    left = totalNodes(tree["left_child"])
    right = totalNodes(tree["right_child"])

    return 1 + left + right

# %%
# calculo da folha com melhor proporção entre positivos e amostras totais para folhas com n >= 200
# a função retorna a folha com n >=200 e com a maior proporção entre positivas e negativas, quantidade de amostras negativas, quantidade de amostras positivas, total de amostras
# e o caminho percorrido até aquela folha (variável, separação (< ou >=) e limiar)
def pathSave(tree, path = None):
    if path is None:
        path = []

    if tree["type"] == "leaf":
        l, r = tree["split"]
        if r >= 200:
            prop = r / (r+l)
            total = l + r 
            return prop, l, r, total, path
        else:
            return (0,0,0,0)
     
    left_path = path + [f"{tree['feature']}: < {tree['threshold']}" ]
    right_path = path + [f"{tree['feature']}: >= {tree['threshold']}" ]

    left = pathSave(tree["left_child"], left_path)
    right = pathSave(tree["right_child"], right_path)

    return max(left, right)

# %%
# cálculo da importância das variáveis em cada árvore
# a importância é calculada pelo decréscimo do gini * a quantidade de amostras no nó 
def importance(tree):
    if tree["type"] == "leaf":
        return []
    
    gini_pai = tree["Gini_parent"]
    gini_no = tree["Gini"]
    profit = gini_pai - gini_no
    l, r, = tree["split"]
    total = l + r 

    lista = [tree["feature"], profit, l, r, total ]

    return [lista] + importance(tree["left_child"]) + importance(tree["right_child"])

# %%
# função que recebe uma amostra e preve em qual folha a mostra seria alocada 
# dentro da folha a probabilidade da amostra ser positiva é calculada através de r / (r + l )
# se r > 0.5 a amostra é tida como positiva 
def predictFunc(tree, sample):

    if tree["type"] == "leaf":
        l, r = tree["split"]
        prob = r / ( l + r )
        return prob
    
    threshold = tree["threshold"]
    feature = tree["feature"]

    if sample[feature] < threshold:
        return predictFunc(tree["left_child"], sample)
    
    if sample[feature] >= threshold:
        return predictFunc(tree["right_child"], sample)

# %%
# função para o cálculo de uma árvore de decisão completa
# percorrendo todos os possíveis limiares para todas a amostras em cada nó 
# deve ser comparada com a random forest 

def full_treeFunc(set, grupo, pos_g):

    gini_pai = ginipai(set, grupo)
    
    if gini_pai ==0:
        return{"type": "leaf", "split":[len(set[set[grupo] != pos_g]), len(set[set[grupo] == pos_g])], "Gini": gini_pai}
    if len(set) < 10: 
        return{"type": "leaf", "split":[len(set[set[grupo] != pos_g]), len(set[set[grupo] == pos_g])], "Gini": gini_pai}

    vars = set.columns.tolist()
    vars.remove(grupo)
    

    gini_no = gini_pai
    threshold = None
    feature = None 
    for i in vars:
        valores =  np.sort(set[i].unique())
        limiares = [(valores[k] + valores[k+1])/2 for k in range(0,len(valores)-1)]
        for j in limiares:
            eval = ginino(set, i, j, grupo, pos_g)
            if eval < gini_no:
                gini_no = eval
                threshold = j 
                feature = i 
    
    if gini_no == gini_pai:
        return{"type": "leaf", "split":[len(set[set[grupo] != pos_g]), len(set[set[grupo] == pos_g])], "Gini": gini_pai} 

    left = set[set[feature] < threshold]
    right = set[set[feature] >= threshold]

    left_child = full_treeFunc(left, grupo, pos_g)
    right_child = full_treeFunc(right, grupo, pos_g)

    node = {"type": "node", "left_child": left_child, "right_child": right_child, 
            "Gini": gini_no, "Gini_parent": gini_pai, "feature": feature,
            "threshold": threshold, "split": [len(left), len(right)]}
    
    return node 


