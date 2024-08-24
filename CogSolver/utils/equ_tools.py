# -*- coding:utf-8 -*-

def pre_solver(pre_equ):
    op_list = set(['+', '-', '/', '*', '^'])
    status = True
    stack = []
    for elem in pre_equ:
        if elem in op_list:
            stack.append((elem, False))
        else:
            if type(elem) is str and '%' in elem:
                elem = float(elem[:-1]) / 100.0
            else:
                elem = float(elem)
            while len(stack) >= 2 and stack[-1][1]:
                opnd = stack.pop()[0]
                op = stack.pop()[0]
                if op == "+":
                    elem = opnd + elem
                elif op == "-":
                    elem = opnd - elem
                elif op == "*":
                    elem = opnd * elem
                elif op == "/":
                    elem = opnd / elem
                elif op == "^":
                    elem = opnd ** elem
                else:
                    status = False
                    break
            if status:
                stack.append((elem, True))
            else:
                break
    if status and len(stack) == 1:
        try:
            answer = float(stack.pop()[0])
        except:
            answer = None
    else:
        answer = None
    return answer
