"""
Este módulo implementa o léxico e o parser da linguagem wec usando PLY.
"""

import ply.lex as lex
import ply.yacc as yacc

# ------------------------------------------------------------
# Tokens e regras léxicas
# ------------------------------------------------------------

# Lista de palavras reservadas
reserved = {
    "module": "MODULE",
    "fn": "FN",
    "let": "LET",
    "return": "RETURN",
    "true": "TRUE",
    "false": "FALSE",
    "if": "IF",
    "else": "ELSE",
    "struct": "STRUCT",
    "enum": "ENUM",
    "join": "JOIN",
    "implement": "IMPLEMENT",
    "uses": "USES",
    "and": "AND",
    "as": "AS",
    "class": "CLASS",
    "new": "NEW",
    "public": "PUBLIC",
    "import": "IMPORT"
}

tokens = [
    "IDENT", "NUMBER", "STRING",
    "ARROW", "DDCOLON",
    "COLON", "SEMI", "COMMA",
    "LPAREN", "RPAREN",
    "LBRACE", "RBRACE",
    "LBRACKET", "RBRACKET",
    "DOT",
    "EQUAL",
    "PLUS", "MINUS", "TIMES", "DIVIDE", "MOD",
    "LE", "LT", "GE", "GT", "EQ", "NE"
] + list(reserved.values())

# Tokens com regex simples
t_ARROW    = r"->"
t_DDCOLON  = r"::"
t_COLON    = r":"
t_SEMI     = r";"
t_COMMA    = r","
t_LPAREN   = r"\("
t_RPAREN   = r"\)"
t_LBRACE   = r"\{"
t_RBRACE   = r"\}"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_DOT      = r"\."
t_EQUAL    = r"="
t_PLUS    = r"\+"
t_MINUS   = r"-"
t_TIMES   = r"\*"
t_DIVIDE  = r"/"
t_LE      = r"<="
t_GE      = r">="
t_EQ      = r"=="
t_NE      = r"!="
t_LT      = r"<"
t_GT      = r">"
t_MOD     = r"%"

t_ignore = " \t\r"

def t_NUMBER(t):
    r"\d+(\.\d+)?"
    if '.' in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    # Remove as aspas e trata escapes simples
    t.value = bytes(t.value[1:-1], "utf-8").decode("unicode_escape")
    return t

def t_IDENT(t):
    r"[A-Za-z_][A-Za-z0-9_]*"
    t.type = reserved.get(t.value, "IDENT")
    return t

def t_newline(t):
    r"\n+"
    t.lexer.lineno += t.value.count("\n")

def t_COMMENT(t):
    r'//.*'
    pass

def t_error(t):
    print("Caractere ilegal: '%s'" % t.value[0])
    t.lexer.skip(1)

lexer = lex.lex()

# ------------------------------------------------------------
# Definição dos nós da AST
# ------------------------------------------------------------

class Program:
    def __init__(self, module, types, functions, type_env=None, imports=None):
        self.module = module  # pode ser None
        self.types = types    # lista de definições de tipos (struct, enum, join, class, etc)
        self.functions = functions
        # novo: ambiente de tipos com mapeamento de t.name para a definição
        self.type_env = type_env or {}
        self.imports = imports or []

class ModuleDecl:
    def __init__(self, name):
        self.name = name

class FunctionDef:
    def __init__(self, name, params, return_type, body):
        self.name = name
        self.params = params   # lista de Parameter
        self.return_type = return_type  # pode ser None
        self.body = body
    def __repr__(self):
        return f"<FunctionDef {self.name}>"

class Parameter:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_
    def __repr__(self):
        return f"Param({self.name}:{self.type})"

class Block:
    def __init__(self, statements):
        self.statements = statements

class VarDecl:
    def __init__(self, name, type_, initializer):
        self.name = name
        self.type = type_
        self.initializer = initializer

class ReturnStmt:
    def __init__(self, expression):
        self.expression = expression

class ExprStmt:
    def __init__(self, expression):
        self.expression = expression

class Identifier:
    def __init__(self, name):
        self.name = name

class Literal:
    def __init__(self, value):
        self.value = value

class ListExpr:
    def __init__(self, elements):
        self.elements = elements

class IndexExpr:
    def __init__(self, receiver, index):
        self.receiver = receiver
        self.index = index

class Arg:
    def __init__(self, name, value):
        self.name = name  # se None, argumento posicional
        self.value = value

class CallExpr:
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args  # lista de Arg

class MemberAccess:
    def __init__(self, receiver, member):
        self.receiver = receiver
        self.member = member

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class IfStmt:
    def __init__(self, condition, then_block, elif_blocks, else_block):
        self.condition = condition
        self.then_block = then_block
        self.elif_blocks = elif_blocks  # lista de tuplas (condição, bloco)
        self.else_block = else_block

class NewExpr:
    def __init__(self, type_name, args):
         self.type_name = type_name
         self.args = args

class Assignment:
    def __init__(self, target, value):
         self.target = target
         self.value = value

class ImportDecl:
    def __init__(self, filename):
         self.filename = filename

# ------------------------------------------------------------
# Regras do Parser
# ------------------------------------------------------------

precedence = (
    ('nonassoc', 'IFX'),
    ('left', 'EQ', 'NE'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'MOD'),
)

def p_program(p):
    """program : opt_module decl_list"""
    # Separa decl_list em definições de importações, tipos e funções.
    types = []
    functions = []
    imports = []
    for d in p[2]:
         if isinstance(d, ImportDecl):
              imports.append(d)
         elif isinstance(d, (StructDef, EnumDef, JoinDef, ImplementDecl, ClassDef)):
              types.append(d)
         else:
              functions.append(d)
    # Cria um ambiente de tipos a partir dos nós que têm atributo "name"
    type_env = {t.name: t for t in types if hasattr(t, "name")}
    p[0] = Program(module=p[1], types=types, functions=functions, type_env=type_env, imports=imports)

def p_opt_module(p):
    """opt_module : module_decl
                  | empty"""
    p[0] = p[1]

def p_decl_list(p):
    """decl_list : decl_list decl
                 | decl"""
    if len(p) == 2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_decl(p):
    """decl : class_def
             | struct_def
             | enum_def
             | join_def
             | implement_decl
             | function
             | import_decl"""
    p[0] = p[1]

def p_module_decl(p):
    "module_decl : MODULE IDENT SEMI"
    p[0] = ModuleDecl(name=p[2])

def p_struct_def(p):
    "struct_def : STRUCT IDENT LBRACE field_decl_list RBRACE"
    p[0] = StructDef(name=p[2], fields=p[4])

def p_enum_def(p):
    "enum_def : ENUM IDENT LBRACE enum_variant_list RBRACE"
    p[0] = EnumDef(name=p[2], variants=p[4])

def p_join_def(p):
    "join_def : JOIN IDENT AND IDENT AS IDENT LBRACE field_decl_list RBRACE"
    p[0] = JoinDef(type1=p[2], type2=p[4], new_type=p[6], fields=p[8])

def p_field_decl_list(p):
    """field_decl_list : field_decl_list field_decl
                        | field_decl
                        | empty"""
    if len(p) == 2:
         p[0] = [] if p[1] is None else [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_field_decl(p):
    "field_decl : IDENT COLON type_expr opt_comma"
    p[0] = (p[1], p[3])

def p_enum_variant_list(p):
    """enum_variant_list : enum_variant_list COMMA IDENT
                         | IDENT
                         | empty"""
    if len(p) == 2:
         p[0] = [] if p[1] is None else [p[1]]
    else:
         p[0] = p[1] + [p[3]]

def p_record_literal(p):
    "primary : IDENT LBRACE field_assign_list RBRACE"
    p[0] = RecordLiteral(type_name=p[1], fields=p[3])

def p_field_assign_list(p):
    """field_assign_list : field_assign_list field_assign
                         | field_assign
                         | empty"""
    if len(p) == 2:
         p[0] = {} if p[1] is None else p[1]
    else:
         d = p[1]
         d.update(p[2])
         p[0] = d

def p_field_assign(p):
    "field_assign : IDENT COLON expression opt_comma"
    p[0] = {p[1]: p[3]}

def p_opt_comma(p):
    """opt_comma : COMMA
                 | empty"""
    p[0] = None

def p_class_def(p):
    "class_def : CLASS IDENT LBRACE class_body RBRACE"
    p[0] = ClassDef(name=p[2], members=p[4])

def p_class_body(p):
    """class_body : class_body class_member
                  | class_member"""
    if len(p) == 2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_class_member_field(p):
    """class_member : PUBLIC IDENT COLON type_expr SEMI
                    | LET IDENT COLON type_expr SEMI"""
    p[0] = ("field", p[2], p[4])

def p_class_member_function(p):
    "class_member : function"
    p[0] = ("method", p[1])

## New AST node definitions for types, record literals, and implementations:
class StructDef:
    def __init__(self, name, fields):
         self.name = name
         self.fields = fields

class EnumDef:
    def __init__(self, name, variants):
         self.name = name
         self.variants = variants

class JoinDef:
    def __init__(self, type1, type2, new_type, fields):
         self.name = new_type  # Set the new type name as the join definition name
         self.type1 = type1
         self.type2 = type2
         self.new_type = new_type
         self.fields = fields

class RecordLiteral:
    def __init__(self, type_name, fields):
         self.type_name = type_name
         self.fields = fields

class ImplementDecl:
    def __init__(self, new_type, base_type, methods):
         self.name = new_type  # new type name, e.g. "Dog"
         self.base_type = base_type  # base type name, e.g. "Animal"
         self.methods = methods      # list of method definitions (FunctionDef)

class ClassDef:
    def __init__(self, name, members):
         self.name = name
         self.members = members  # List of ("field", fieldName, type) or ("method", FunctionDef)

def p_function(p):
    "function : FN IDENT LPAREN parameters_opt RPAREN return_opt block"
    p[0] = FunctionDef(name=p[2], params=p[4], return_type=p[6], body=p[7])

def p_parameters_opt(p):
    """parameters_opt : parameters
                      | empty"""
    p[0] = p[1] if p[1] is not None else []

def p_parameters(p):
    """parameters : parameters COMMA parameter
                  | parameter"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_parameter(p):
    "parameter : IDENT COLON type_expr"
    p[0] = Parameter(name=p[1], type_=p[3])

def p_return_opt(p):
    """return_opt : ARROW type_expr
                  | empty"""
    p[0] = p[2] if len(p) > 2 else None

def p_type_expr(p):
    '''type_expr : simple_type_expr array_spec_opt'''
    p[0] = p[1] + p[2]

def p_simple_type_expr(p):
    '''simple_type_expr : IDENT type_access_opt'''
    p[0] = p[1] + (p[2] if p[2] is not None else "")

def p_array_spec_opt(p):
    '''array_spec_opt : LBRACKET NUMBER RBRACKET
                      | empty'''
    if len(p) == 4:
         p[0] = f"[{p[2]}]"
    else:
         p[0] = ""

def p_type_access_opt(p):
    """type_access_opt : type_access_opt_part
                       | empty"""
    p[0] = p[1]

def p_type_access_opt_part(p):
    """type_access_opt_part : DOT IDENT
                            | DDCOLON IDENT"""
    p[0] = p[1] + p[2]

def p_block(p):
    "block : LBRACE statement_list RBRACE"
    p[0] = Block(statements=p[2])

def p_statement_list(p):
    """statement_list : statement_list statement
                      | empty"""
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = []

def p_statement(p):
    """statement : var_decl
                 | assignment_statement
                 | return_statement
                 | expr_statement
                 | if_statement"""
    p[0] = p[1]

def p_var_decl(p):
    """var_decl : LET IDENT COLON type_expr EQUAL expression SEMI
                | LET IDENT COLON type_expr SEMI"""
    if len(p) == 8:
         p[0] = VarDecl(name=p[2], type_=p[4], initializer=p[6])
    else:
         p[0] = VarDecl(name=p[2], type_=p[4], initializer=None)

def p_return_statement(p):
    "return_statement : RETURN expression SEMI"
    p[0] = ReturnStmt(expression=p[2])

def p_expr_statement(p):
    "expr_statement : expression SEMI"
    p[0] = ExprStmt(expression=p[1])

def p_expression(p):
    """expression : call_expr
                  | expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression MOD expression
                  | expression LE expression
                  | expression LT expression
                  | expression GE expression
                  | expression GT expression
                  | expression EQ expression
                  | expression NE expression"""
    if len(p) == 2:
         p[0] = p[1]
    else:
         p[0] = BinOp(left=p[1], op=p[2], right=p[3])

def p_call_expr(p):
    "call_expr : call_expr call_suffix"
    # Se o sufixo for "call", encapsula em CallExpr;
    # se for "index", cria uma expressão de indexação;
    # Senão, trata como acesso a membro.
    if isinstance(p[2], tuple):
        if p[2][0] == "call":
             p[0] = CallExpr(callee=p[1], args=p[2][1])
        elif p[2][0] == "index":
             p[0] = IndexExpr(receiver=p[1], index=p[2][1])
        else:
             p[0] = MemberAccess(receiver=p[1], member=p[2])
    else:
        p[0] = MemberAccess(receiver=p[1], member=p[2])

def p_call_expr_base(p):
    "call_expr : primary"
    p[0] = p[1]

def p_call_suffix_call(p):
    "call_suffix : LPAREN arg_list_opt RPAREN"
    p[0] = ("call", p[2])

def p_call_suffix_dot(p):
    "call_suffix : DOT IDENT"
    p[0] = p[2]

def p_call_suffix_ddcolon(p):
    "call_suffix : DDCOLON IDENT"
    p[0] = p[2]

def p_call_suffix_arrow(p):
    "call_suffix : ARROW IDENT"
    p[0] = p[2]

def p_call_suffix_index(p):
    "call_suffix : LBRACKET expression RBRACKET"
    p[0] = ("index", p[2])

def p_arg_list_opt(p):
    """arg_list_opt : arg_list
                    | empty"""
    p[0] = p[1] if p[1] is not None else []

def p_arg_list(p):
    """arg_list : arg_list COMMA argument
                | argument"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_argument(p):
    """argument : IDENT COLON expression
                | expression"""
    if len(p) == 4:
        p[0] = Arg(name=p[1], value=p[3])
    else:
        p[0] = Arg(name=None, value=p[1])

def p_primary_paren(p):
    "primary : LPAREN expression RPAREN"
    p[0] = p[2]

def p_primary_list(p):
    "primary : LBRACKET expr_list_opt RBRACKET"
    p[0] = ListExpr(elements=p[2])

def p_expr_list_opt(p):
    """expr_list_opt : expr_list
                     | empty"""
    p[0] = p[1] if p[1] is not None else []

def p_expr_list(p):
    """expr_list : expr_list COMMA expression
                 | expression"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_primary_literal_number(p):
    "primary : NUMBER"
    p[0] = Literal(value=p[1])

def p_primary_literal_string(p):
    "primary : STRING"
    p[0] = Literal(value=p[1])

def p_primary_identifier(p):
    "primary : IDENT"
    p[0] = Identifier(name=p[1])

def p_primary_boolean(p):
    """primary : TRUE
               | FALSE"""
    value = True if str(p[1]).lower() == "true" else False
    p[0] = Literal(value=value)

def p_primary_new(p):
    """primary : NEW IDENT LPAREN arg_list_opt RPAREN
                | NEW IDENT"""
    if len(p) == 3:
         p[0] = NewExpr(type_name=p[2], args=[])
    else:
         p[0] = NewExpr(type_name=p[2], args=p[4])

def p_primary_lambda(p):
    "primary : FN LPAREN parameters_opt RPAREN return_opt block"
    p[0] = FunctionDef(name=None, params=p[3], return_type=p[5], body=p[6])

def p_empty(p):
    "empty :"
    p[0] = None

def p_if_statement(p):
    """if_statement : IF expression block else_clause_opt %prec IFX"""
    if p[4] is None:
        p[0] = IfStmt(condition=p[2], then_block=p[3], elif_blocks=[], else_block=None)
    else:
        p[0] = IfStmt(condition=p[2], then_block=p[3], elif_blocks=[], else_block=p[4])

def p_else_clause_opt(p):
    """else_clause_opt : ELSE if_statement
                        | ELSE block
                        | empty"""
    if len(p) == 1:
        p[0] = None
    else:
        p[0] = p[2]

def p_implement_decl(p):
    "implement_decl : IMPLEMENT IDENT USES IDENT LBRACE method_list RBRACE"
    p[0] = ImplementDecl(new_type=p[2], base_type=p[4], methods=p[6])

def p_method_list(p):
    """method_list : method_list function
                   | function"""
    if len(p) == 2:
         p[0] = [p[1]]
    else:
         p[0] = p[1] + [p[2]]

def p_assignment_statement(p):
    "assignment_statement : assignment SEMI"
    p[0] = p[1]

def p_assignment(p):
    "assignment : call_expr EQUAL expression"
    p[0] = Assignment(target=p[1], value=p[3])

def p_import_decl(p):
    "import_decl : IMPORT STRING SEMI"
    p[0] = ImportDecl(filename=p[2])

def p_error(p):
    if p:
        print("Erro sintático no token", p.type, "com valor", p.value)
    else:
        print("Erro sintático: EOF")

parser = yacc.yacc()

def parse_wec(source):
    return parser.parse(source, lexer=lexer)
