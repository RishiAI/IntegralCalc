```mermaid
graph TD
    A[Start] --> B[User Input]
    B --> C[IntegralQueryParser.parse_query]
    C --> D["Tokenization (using re.finditer)"]
    D --> E["Parse Query (extract integral type, function, bounds)"]

    E --> F{"Determine Integral Type\n(using keywords in query)"}
    F -->|"'riemann' in query"| G["Set type = 'standard'"]
    F -->|"'lebesgue' in query"| H["Set type = 'lebesgue'"]
    F -->|"'ito' in query"| I["Set type = 'darboux'"]
    F -->|"Default"| G

    E --> J["Extract Function\n(using regex patterns)"]
    J --> K{"Is Dirichlet Function?\n(regex match)"}
    K -->|Yes| L["Return '1 if is_rational(x) else 0'"]
    K -->|No| M[MathParser.parse]

    M --> N["Tokenize (MathLexer.tokenize)"]
    N --> O["Build AST (recursive parsing)"]
    O --> P["Convert AST to string\n(ast_to_string method)"]

    E --> Q["Extract Bounds\n(using regex)"]

    L --> R["Compile parsed info into dict"]
    P --> R
    Q --> R
    G --> R
    H --> R
    I --> R

    R --> S["compute_integral(func_str, a, b, n=1000, r=100)"]
    S --> T{"Is Dirichlet Function?\n(func_str == '1 if is_rational(x) else 0')"}
    T -->|Yes| U["Use inte.lebesgue(func, a, b, r, n)"]
    T -->|No| V["determine_best_integral(func, a, b)"]

    V --> W{"is_differentiable(f, a, b)"}
    W -->|Yes| X["Use inte.riemann(func, a, b, n)"]
    W -->|No| Y{"is_continuous(f, a, b)"}
    Y -->|Yes| Z["Use inte.darboux(func, a, b, n)"]
    Y -->|No| AA["Use inte.lebesgue(func, a, b, r, n)"]

    U --> AB["Calculate Integral\n(call appropriate inte.* function)"]
    X --> AB
    Z --> AB
    AA --> AB

    AB --> AC["Return (result, method)"]
    AC --> AD[End]

    subgraph MathParser
    N["def tokenize(text):\n    return list of Token objects"]
    O["def expression(), term(), factor():\n    recursively build AST"]
    P["def ast_to_string(node):\n    convert AST to function string"]
    end

    subgraph IntegralTypeDetection
    F["if-elif statements checking for keywords"]
    end

    subgraph FunctionExtraction
    J["regex: r'f\s*\(\s*x\s*\)\s*=\s*(.+?)\s+from'"]
    K["regex: r'f\s*\(\s*x\s*\)\s*=\s*1\s+for\s+rational'"]
    end

    subgraph IntegrationMethodSelection
    T["if statement for Dirichlet"]
    V["def determine_best_integral(f, a, b):"]
    W["def is_differentiable(f, a, b, h=1e-5, n=1000):"]
    Y["def is_continuous(f, a, b, n=1000):"]
    end

    class A,AD nodeStyle;
    classDef nodeStyle fill:#f9f,stroke:#333,stroke-width:2px;
