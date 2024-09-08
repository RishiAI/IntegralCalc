from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Dict, Callable
import math
import matplotlib.pyplot as plt

@dataclass
class Symbol:
    name: str

@dataclass
class Number:
    value: float

@dataclass
class Expression:
    operator: str
    operands: List[Union[Expression, Symbol, Number]]

class AdvancedCAS:
    def __init__(self):
        self.variables: Dict[str, float] = {}
        self.functions: Dict[str, Callable] = {
            'sin': math.sin,
            'cos': math.cos,
            'exp': math.exp,
            'log': math.log,
            'sqrt': math.sqrt
        }

    def simplify(self, expr: Union[Expression, Symbol, Number]) -> Union[Expression, Symbol, Number]:
        if isinstance(expr, (Symbol, Number)):
            return expr
        
        simplified_operands = [self.simplify(op) for op in expr.operands]
        
        if all(isinstance(op, Number) for op in simplified_operands):
            if expr.operator == '+':
                return Number(sum(op.value for op in simplified_operands))
            elif expr.operator == '*':
                return Number(math.prod(op.value for op in simplified_operands))
            elif expr.operator == '-' and len(simplified_operands) == 2:
                return Number(simplified_operands[0].value - simplified_operands[1].value)
            elif expr.operator == '/' and len(simplified_operands) == 2:
                return Number(simplified_operands[0].value / simplified_operands[1].value)
            elif expr.operator == '^' and len(simplified_operands) == 2:
                return Number(simplified_operands[0].value ** simplified_operands[1].value)
        
        # Combine like terms
        if expr.operator == '+':
            terms = {}
            constant = 0
            for op in simplified_operands:
                if isinstance(op, Number):
                    constant += op.value
                elif isinstance(op, Symbol):
                    terms[op.name] = terms.get(op.name, 0) + 1
                elif isinstance(op, Expression) and op.operator == '*':
                    coef = 1
                    var = None
                    for term in op.operands:
                        if isinstance(term, Number):
                            coef *= term.value
                        elif isinstance(term, Symbol):
                            var = term.name
                        elif isinstance(term, Expression):
                            var = self.to_string(term)
                    if var:
                        terms[var] = terms.get(var, 0) + coef
                    else:
                        constant += coef
                else:
                    # Handle other types of expressions
                    var = self.to_string(op)
                    terms[var] = terms.get(var, 0) + 1
            
            new_operands = []
            for var, coef in terms.items():
                if coef == 1:
                    new_operands.append(Symbol(var) if len(var) == 1 else self.parse_string(var))
                elif coef != 0:
                    new_operands.append(Expression('*', [Number(coef), Symbol(var) if len(var) == 1 else self.parse_string(var)]))
            
            if constant != 0 or not new_operands:
                new_operands.append(Number(constant))
            
            if len(new_operands) == 1:
                return new_operands[0]
            else:
                return Expression('+', sorted(new_operands, key=lambda x: self.sort_key(x)))
        
        return Expression(expr.operator, simplified_operands)

    def sort_key(self, expr):
        if isinstance(expr, Number):
            return (0, expr.value)
        elif isinstance(expr, Symbol):
            return (1, expr.name)
        elif isinstance(expr, Expression):
            if expr.operator == '*' and len(expr.operands) == 2 and isinstance(expr.operands[0], Number):
                return (2, self.sort_key(expr.operands[1]))
            else:
                return (3, self.to_string(expr))

    def factor(self, expr: Expression) -> Expression:
        simplified = self.simplify(expr)
        if not isinstance(simplified, Expression) or simplified.operator != '+':
            return simplified
        
        terms = {0: 0, 1: 0, 2: 0}
        for term in simplified.operands:
            if isinstance(term, Number):
                terms[0] += term.value
            elif isinstance(term, Symbol):
                terms[1] += 1
            elif isinstance(term, Expression) and term.operator == '*':
                coef = 1
                power = 1
                for factor in term.operands:
                    if isinstance(factor, Number):
                        coef *= factor.value
                    elif isinstance(factor, Expression) and factor.operator == '^':
                        power = factor.operands[1].value
                terms[power] += coef
        
        a, b, c = terms[2], terms[1], terms[0]
        if a == 0:
            return simplified
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return simplified  # Cannot factor with complex roots
        elif discriminant == 0:
            root = -b / (2*a)
            if root == 0:
                return Expression('*', [Number(a), Expression('^', [Symbol('x'), Number(2)])])
            else:
                return Expression('*', [
                    Number(a),
                    Expression('^', [
                        Expression('+', [Symbol('x'), Number(-root)]),
                        Number(2)
                    ])
                ])
        else:
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)
            if a == 1:
                return Expression('*', [
                    Expression('+', [Symbol('x'), Number(-root1)]),
                    Expression('+', [Symbol('x'), Number(-root2)])
                ])
            else:
                return Expression('*', [
                    Number(a),
                    Expression('+', [Symbol('x'), Number(-root1)]),
                    Expression('+', [Symbol('x'), Number(-root2)])
                ])

    def complete_square(self, expr: Expression) -> Expression:
        simplified = self.simplify(expr)
        if not isinstance(simplified, Expression) or simplified.operator != '+':
            return simplified
        
        terms = {0: 0, 1: 0, 2: 0}
        for term in simplified.operands:
            if isinstance(term, Number):
                terms[0] += term.value
            elif isinstance(term, Symbol):
                terms[1] += 1
            elif isinstance(term, Expression) and term.operator == '*':
                coef = 1
                power = 1
                for factor in term.operands:
                    if isinstance(factor, Number):
                        coef *= factor.value
                    elif isinstance(factor, Expression) and factor.operator == '^':
                        power = factor.operands[1].value
                terms[power] += coef
        
        a, b, c = terms[2], terms[1], terms[0]
        if a == 0:
            return simplified  # Not a quadratic expression
        
        h = -b / (2*a)
        k = c - (b**2) / (4*a)
        
        if a == 1:
            return Expression('+', [
                Expression('^', [
                    Expression('+', [Symbol('x'), Number(-h)]),
                    Number(2)
                ]),
                Number(k)
            ])
        else:
            return Expression('+', [
                Expression('*', [
                    Number(a),
                    Expression('^', [
                        Expression('+', [Symbol('x'), Number(-h)]),
                        Number(2)
                    ])
                ]),
                Number(k)
            ])

    def graph(self, expr: Expression, x_range: tuple[float, float], points: int = 100):
        x_values = [x_range[0] + (x_range[1] - x_range[0]) * i / points for i in range(points + 1)]
        y_values = [self.evaluate(expr, {'x': x}) for x in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values)
        plt.title(f"Graph of {self.to_string(expr)}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()

    def evaluate(self, expr: Union[Expression, Symbol, Number], values: Dict[str, float]) -> float:
        if isinstance(expr, Number):
            return expr.value
        elif isinstance(expr, Symbol):
            return values.get(expr.name, 0)
        elif isinstance(expr, Expression):
            operands = [self.evaluate(op, values) for op in expr.operands]
            if expr.operator in self.functions:
                return self.functions[expr.operator](*operands)
            elif expr.operator == '+':
                return sum(operands)
            elif expr.operator == '*':
                return math.prod(operands)
            elif expr.operator == '-' and len(operands) == 2:
                return operands[0] - operands[1]
            elif expr.operator == '/' and len(operands) == 2:
                return operands[0] / operands[1]
            elif expr.operator == '^' and len(operands) == 2:
                return operands[0] ** operands[1]
        raise ValueError(f"Unable to evaluate expression: {expr}")

    def to_string(self, expr: Union[Expression, Symbol, Number]) -> str:
        if isinstance(expr, Number):
            return f"{expr.value:.2f}" if expr.value % 1 else f"{int(expr.value)}"
        elif isinstance(expr, Symbol):
            return expr.name
        elif isinstance(expr, Expression):
            if expr.operator in self.functions:
                return f"{expr.operator}({', '.join(self.to_string(op) for op in expr.operands)})"
            elif expr.operator == '+':
                return ' + '.join(self.to_string(op) for op in expr.operands)
            elif expr.operator == '-':
                if len(expr.operands) == 1:
                    return f"-{self.to_string(expr.operands[0])}"
                else:
                    return f"{self.to_string(expr.operands[0])} - {self.to_string(expr.operands[1])}"
            elif expr.operator == '*':
                return ' · '.join(self.parenthesize_if_needed(op) for op in expr.operands)
            elif expr.operator == '/':
                return f"{self.parenthesize_if_needed(expr.operands[0])} / {self.parenthesize_if_needed(expr.operands[1])}"
            elif expr.operator == '^':
                return f"{self.parenthesize_if_needed(expr.operands[0])}^{self.parenthesize_if_needed(expr.operands[1])}"
        raise ValueError(f"Unable to convert to string: {expr}")

    def parenthesize_if_needed(self, expr: Union[Expression, Symbol, Number]) -> str:
        s = self.to_string(expr)
        if isinstance(expr, Expression) and expr.operator in ['+', '-']:
            return f"({s})"
        return s

    def parse_string(self, s: str) -> Union[Expression, Symbol, Number]:
        # This is a very basic parser and would need to be expanded for a full CAS
        if '^' in s:
            base, exp = s.split('^')
            return Expression('^', [self.parse_string(base), self.parse_string(exp)])
        elif '*' in s:
            return Expression('*', [self.parse_string(term) for term in s.split('*')])
        elif s.isalpha():
            return Symbol(s)
        else:
            try:
                return Number(float(s))
            except ValueError:
                raise ValueError(f"Unable to parse: {s}")

# Example usage
if __name__ == "__main__":
    cas = AdvancedCAS()
    x = Symbol('x')

    # Test simplification
    expr1 = Expression('+', [Expression('*', [Number(2), x]), Expression('*', [Number(3), x]), Number(5)])
    simplified1 = cas.simplify(expr1)
    print(f"Simplified: {cas.to_string(simplified1)}")

    # Test factoring
    expr2 = Expression('+', [
        Expression('^', [x, Number(2)]),
        Expression('*', [Number(5), x]),
        Number(6)
    ])
    factored = cas.factor(expr2)
    print(f"Factored: {cas.to_string(factored)}")

    # Test completing the square
    completed_square = cas.complete_square(expr2)
    print(f"Completed square: {cas.to_string(completed_square)}")

    # Test more complex expressions
    expr3 = Expression('+', [
        Expression('*', [Number(2), Expression('^', [x, Number(3)])]),
        Expression('*', [Number(-3), Expression('^', [x, Number(2)])]),
        Expression('*', [Number(4), x]),
        Number(-5)
    ])
    simplified3 = cas.simplify(expr3)
    print(f"Complex expression: {cas.to_string(expr3)}")
    print(f"Simplified complex: {cas.to_string(simplified3)}")

    # Test graphing
    cas.graph(expr2, (-5, 5))