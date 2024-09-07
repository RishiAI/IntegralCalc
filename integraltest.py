import re
import math
import numpy as np
import integrals as inte

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

class MathLexer:
    def __init__(self):
        self.tokens = [
            ('NUMBER', r'\d+(\.\d*)?'),
            ('FUNCTION', r'(sin|cos|tan|exp|log|sqrt|is_rational)'),
            ('OPERATOR', r'[\+\-\*/\^]'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('CONDITIONAL', r'if|else'),
            ('COMPARISON', r'==|!=|<=|>=|<|>'),
            ('VARIABLE', r'[a-zA-Z]'),
        ]
        self.token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.tokens)

    def tokenize(self, text):
        return [Token(match.lastgroup, match.group()) 
                for match in re.finditer(self.token_regex, text) 
                if match.lastgroup != 'WHITESPACE']

class ASTNode:
    def __init__(self, token):
        self.token = token
        self.left = None
        self.right = None


class MathParser:
    def __init__(self):
        self.lexer = MathLexer()

    def parse(self, text):
        self.tokens = self.lexer.tokenize(text)
        self.current_token = 0
        return self.expression()

    def expression(self):
        node = self.term()
        while self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'OPERATOR':
            token = self.tokens[self.current_token]
            self.current_token += 1
            right = self.term()
            new_node = ASTNode(token)
            new_node.left = node
            new_node.right = right
            node = new_node
        return node

    def term(self):
        if self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'OPERATOR' and self.tokens[self.current_token].value in ['+', '-']:
            # handle unary operators
            token = self.tokens[self.current_token]
            self.current_token += 1
            operand = self.term()
            node = ASTNode(token)
            node.left = ASTNode(Token('NUMBER', '0'))  # add 0 for unary plus/minus
            node.right = operand
            return node
        elif self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'FUNCTION':
            node = ASTNode(self.tokens[self.current_token])
            self.current_token += 1
            if self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'LPAREN':
                self.current_token += 1
                node.left = self.expression()
                if self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'RPAREN':
                    self.current_token += 1
                else:
                    raise ValueError("Expected closing parenthesis")
            return node
        elif self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'LPAREN':
            self.current_token += 1
            node = self.expression()
            if self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'RPAREN':
                self.current_token += 1
                return node
            else:
                raise ValueError("Expected closing parenthesis")
        elif self.current_token < len(self.tokens) and self.tokens[self.current_token].type in ['NUMBER', 'VARIABLE']:
            node = ASTNode(self.tokens[self.current_token])
            self.current_token += 1
            return node
        elif self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'CONDITIONAL':
            return self.parse_conditional()
        else:
            raise ValueError(f"Unexpected token: {self.tokens[self.current_token].value if self.current_token < len(self.tokens) else 'EOF'}")

    def parse_conditional(self):
        if_node = ASTNode(self.tokens[self.current_token])
        self.current_token += 1
        if_node.left = self.expression()
        if self.current_token < len(self.tokens) and self.tokens[self.current_token].type == 'CONDITIONAL' and self.tokens[self.current_token].value == 'else':
            self.current_token += 1
            if_node.right = self.expression()
        return if_node

    def ast_to_string(self, node):
        if node.token.type == 'NUMBER':
            return node.token.value
        elif node.token.type == 'VARIABLE':
            return 'x'
        elif node.token.type == 'FUNCTION':
            if node.token.value == 'is_rational':
                return f"is_rational({self.ast_to_string(node.left)})"
            return f"math.{node.token.value}({self.ast_to_string(node.left)})"
        elif node.token.type == 'OPERATOR':
            left = self.ast_to_string(node.left)
            right = self.ast_to_string(node.right)
            if node.token.value == '^':
                return f"({left}**{right})"
            return f"({left}{node.token.value}{right})"
        elif node.token.type == 'CONDITIONAL':
            condition = self.ast_to_string(node.left)
            true_value = '1'
            false_value = '0'
            if node.right:
                false_value = self.ast_to_string(node.right)
            return f"({true_value} if {condition} else {false_value})"
        else:
            raise ValueError(f"Unexpected node type: {node.token.type}")

class IntegralQueryParser:
    def __init__(self):
        self.math_parser = MathParser()

    def parse_query(self, query):
        query = query.lower()
        
        integral_type = self.determine_integral_type(query)
        function = self.extract_function(query)
        lower_bound, upper_bound = self.extract_bounds(query)
        
        return {
            'integral_type': integral_type,
            'function': function,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def determine_integral_type(self, query):
        if any(word in query for word in ['riemann', 'standard', 'regular', 'normal']):
            return 'standard'
        elif any(word in query for word in ['lebesgue', 'measure']):
            return 'lebesgue'
        elif any(word in query for word in ['ito', 'itô', 'stochastic', 'brownian']):
            return 'ito'
        return 'standard'  # default to standard if not specified

    def extract_function(self, query):
        # check for dirichlet function description
        dirichlet_pattern = r'f\s*\(\s*x\s*\)\s*=\s*1\s+for\s+rational.*?0\s+for\s+irrational'
        if re.search(dirichlet_pattern, query, re.IGNORECASE):
            return "1 if is_rational(x) else 0"
        
        # check for general function description
        function_match = re.search(r'f\s*\(\s*x\s*\)\s*=\s*(.+?)(?:\s+from|$)', query)
        if function_match:
            ast = self.math_parser.parse(function_match.group(1))
            return self.math_parser.ast_to_string(ast)
        
        # fallback to simpler extraction
        function_match = re.search(r'(?:of|is)\s+(.+?)(?:\s+from|$)', query)
        if function_match:
            ast = self.math_parser.parse(function_match.group(1))
            return self.math_parser.ast_to_string(ast)
        
        return None

    def extract_bounds(self, query):
        bounds_match = re.search(r'from\s*([-]?\d*\.?\d+)\s*to\s*([-]?\d*\.?\d+)', query)
        if bounds_match:
            return float(bounds_match.group(1)), float(bounds_match.group(2))
        return None, None


def is_rational(x, tolerance=1e-10):
    return abs(x - round(x)) < tolerance

def parse_function(func_str):
    safe_dict = {
        "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "pi": math.pi, "e": math.e,
        "sqrt": math.sqrt, "is_rational": is_rational
    }
    
    def func(x):
        safe_dict['x'] = x
        return eval(func_str, {"__builtins__": None}, safe_dict)
    
    return func

def is_continuous(f, a, b, n=1000):
    x = np.linspace(a, b, n)
    y = np.array([f(xi) for xi in x])
    return np.all(np.isfinite(y))

def is_differentiable(f, a, b, h=1e-5, n=1000):
    x = np.linspace(a, b, n)
    try:
        derivative = [(f(xi + h) - f(xi)) / h for xi in x]
        return np.all(np.isfinite(derivative))
    except:
        return False

def determine_best_integral(f, a, b):
    if is_differentiable(f, a, b):
        return "riemann"
    elif is_continuous(f, a, b):
        return "darboux"
    else:
        return "lebesgue"

def compute_integral(func_str, a, b, n=1000, r=100):
    # check for dirichlet function first
    if func_str == "1 if is_rational(x) else 0":
        return inte.lebesgue(parse_function(func_str), a, b, r, n), "lebesgue"
    
    # for other functions, determine the best method
    func = parse_function(func_str)
    integral_type = determine_best_integral(func, a, b)
    
    if integral_type == "riemann":
        result = inte.riemann(func, a, b, n)
    elif integral_type == "darboux":
        result = inte.darboux(func, a, b, n)
    else:  
        result = inte.lebesgue(func, a, b, r, n)
    
    return result, integral_type

def user_interface():
    print("Advanced Mathematical Expression Parser with Improved Dirichlet Function Parsing")
    print("You can now ask questions in natural language.")
    print("For example:")
    print("- 'Calculate the integral of x^2 from 0 to 1'")
    print("- 'Compute the Lebesgue integral of f(x)=1 for rational x and f(x)=0 for irrational x from 0 to 1'")
    print("Type 'quit' to exit.")
    
    query_parser = IntegralQueryParser()
    
    while True:
        user_input = input("\nWhat would you like to calculate? ")
        
        if user_input.lower() == 'quit':
            break
        
        try:
            intent = query_parser.parse_query(user_input)
            
            if intent['function'] is None:
                print("I'm sorry, I couldn't understand the function you want to integrate.")
                continue
            
            if intent['lower_bound'] is None or intent['upper_bound'] is None:
                print("I'm sorry, I couldn't determine the bounds of integration.")
                continue
            
            print(f"\nI understand you want to calculate a {intent['integral_type']} integral.")
            print(f"Function: {intent['function']}")
            print(f"Bounds: from {intent['lower_bound']} to {intent['upper_bound']}")
            
            result, method = compute_integral(intent['function'], intent['lower_bound'], intent['upper_bound'])
            print(f"\nThe integral of the given function from {intent['lower_bound']} to {intent['upper_bound']} is approximately {result}")
            print(f"Method used: {method}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try rephrasing your input.")

if __name__ == "__main__":
    user_interface()
