"""
Este módulo implementa um interpretador para a linguagem wec.
Utiliza os nós da AST definidos em parser_lexer.py e executa
o código interpretado, realizando verificações de tipo simples.
"""

from parser_lexer import (
    Program, FunctionDef, VarDecl, ReturnStmt, ExprStmt, Block, CallExpr, MemberAccess,
    Literal, Identifier, ListExpr, Arg, IfStmt, BinOp, StructDef, EnumDef, JoinDef,
    RecordLiteral, ClassDef, NewExpr, Assignment, UnaryOp, WhileStmt, ForStmt,
    PythonImport, PyFnDecl, PyTypeDecl, parse_wec
)
import re
import math
import os
import sys

# NEW: Global flag for compiled mode (set via wec.py -c) and a setter function.
COMPILED_MODE = False

def set_compiled_mode(mode):
    global COMPILED_MODE
    COMPILED_MODE = mode

# Exceção para controlar a instrução return
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# Classe para o ambiente (escopo) do interpretador
class Environment:
    def __init__(self, outer=None):
        self.outer = outer
        self.values = {}
    def get(self, key):
        if key in self.values:
            return self.values[key]
        elif self.outer:
            return self.outer.get(key)
        else:
            # ADDED: Provide file and line number info for easier debugging.
            import traceback
            tb = traceback.extract_stack()[-2]  # get caller's info
            raise Exception(f"Undefined variable: {key} (at {tb.filename}, line {tb.lineno})")
    def set(self, key, value):
        self.values[key] = value
    def get_type(self, key):
        return self.get(key)

class EnumValue:
    def __init__(self, enum_def, variant):
        self.enum_def = enum_def
        self.variant = variant
        self.wec_type = enum_def.name  # make the type equal to the enum name

    def to_int(self):
        # Return the zero-based index of the variant in the enum's variant list
        return self.enum_def.variants.index(self.variant)

    def __repr__(self):
        return f"{self.enum_def.name}::{self.variant}"

# Função auxiliar para identificar o "tipo" de um valor
def get_type(value):
    if isinstance(value, dict) and "wec_type" in value:
        return value["wec_type"]
    elif hasattr(value, "wec_type"):
        return value.wec_type
    elif isinstance(value, bool):  # Check booleans before integers
        return "bool"
    elif isinstance(value, int):
        return "i32"
    elif isinstance(value, float):
        return "f32"
    elif isinstance(value, str):
        return "str"
    elif isinstance(value, bytes):
        return "bytes"
    elif isinstance(value, list):
        if value:
            etype = get_type(value[0])
            return f"{etype}Array"
        else:
            return "EmptyArray"
    elif hasattr(value, "__fspath__"):  # Verifica se é PathLike
        return "PathLike"
    else:
        return type(value).__name__

def type_matches(declared_type, actual_type):
    # Se o tipo declarado for sysos::Path, aceita str, bytes ou os.PathLike
    if declared_type == "sysos::Path":
        return actual_type in ("str", "bytes") or "PathLike" in actual_type
    # Exact match or if declared is like "i32Array[5]" and actual is "i32Array"
    if declared_type == actual_type:
        return True
    m = re.match(r"([0-9a-zA-Z_]+Array)\[\d+\]", declared_type)
    if m and m.group(1) == actual_type:
        return True
    return False

# ---------------------
# Implementação de Builtins
# ---------------------

# wec.print
def wec_print(*args, **kwargs):
    print(*args, **kwargs)

def wec_input(prompt):
    return input(prompt)

def wec_randomInt(a, b):
    import random
    return random.randint(a, b)

def wec_randomFloat(a, b):
    import random
    return random.uniform(a, b)

def wec_randomString(*args):
    import random
    if args:
        return random.choice(args)
    return ""

def wec_array_to_string(arr):
    return str(arr)

def wec_join_str(*args):
    return "".join(str(arg) for arg in args)

def wec_join_int(*args):
    s = "".join(str(int(arg)) for arg in args)
    try:
        return int(s)
    except:
        raise Exception("wec.join_int: cannot convert " + s + " to int")

def wec_join_float(*args):
    if len(args) == 0:
        return 0.0
    # For join_float, join the integer parts of all arguments
    # and then join the fractional parts (from 2nd arg onward)
    whole = "".join(str(int(arg)) for arg in args)
    frac = "".join((str(arg).split(".")[1] if "." in str(arg) else "") for arg in args[1:])
    if frac:
        result_str = whole + "." + frac
    else:
        result_str = whole
    try:
        return float(result_str)
    except:
        raise Exception("wec.join_float: error converting " + result_str + " to float")

def wec_type_of(val):
    return get_type(val)

def wec_exp(val):
    # Se o valor for uma matriz, aplicar math.exp em cada elemento.
    if isinstance(val, WECMatrix):
         result = WECMatrix()
         result.to(val.dimensions)
         result.wec_type = val.wec_type
         for row in val:
             new_row = WECList()
             new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
             for element in row:
                 new_row.append(math.exp(element))
             result.append(new_row)
         return result
    elif isinstance(val, WECList):
         new_list = WECList()
         new_list.wec_type = val.wec_type if hasattr(val, "wec_type") else "i32List"
         for element in val:
             new_list.append(math.exp(element))
         return new_list
    else:
         return math.exp(val)

# Add built-in file support

class WECFile:
    # Now accepts optional parameters so that we can later call open().
    def __init__(self, filename=None, mode=None):
        self.filename = filename
        self.mode = mode
        self.handle = None
        # Only auto-open if both filename and mode were provided.
        if filename is not None and mode is not None:
            if mode in ("w", "a"):
                self.handle = open(filename, mode)

    # Added an open() method to allow opening a file after construction.
    def open(self, filename, mode):
        self.filename = filename
        self.mode = mode
        if mode in ("w", "a"):
            self.handle = open(filename, mode)

    def write(self, text):
        if self.mode not in ("w", "a"):
            raise Exception("File not open for writing")
        if self.handle is None:
            self.handle = open(self.filename, self.mode)
        self.handle.write(text)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def read(self):
        with open(self.filename, "r") as f:
            return f.read()

class WECFileBuiltin:
    builtin_file = True
    name = "File"
    # Use *args: if no argument, create an empty file instance; if two arguments, use them.
    def new_instance(self, *args):
        if len(args) == 0:
            return WECFile()
        elif len(args) == 2:
            return WECFile(args[0], args[1])
        else:
            raise Exception("File constructor requires 0 or 2 arguments: filename and mode")
    def __call__(self, *args):
        return self.new_instance(*args)

def wec_range(start, end):
    return range(start, end)

# ---------------------
# Classe Interpretador
# ---------------------
class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self.program = None
        self._add_builtins(self.global_env)  # Basic builtins only
    
    def _add_builtins(self, env):
        # Cria o módulo wec com os builtins
        # "tf" será um objeto LazyTensorFlow que só importa tensorflow sob demanda
        wec_module = {
            "print": wec_print,
            "input": wec_input,
            "randomInt": wec_randomInt,
            "randomFloat": wec_randomFloat,
            "randomString": wec_randomString,
            "Array": {"to_string": wec_array_to_string},
            "join_str": wec_join_str,
            "join_int": wec_join_int,
            "join_float": wec_join_float,
            "type_of": wec_type_of,
            "exp": wec_exp,
            "File": WECFileBuiltin(),  # Register the built-in File type.
            "range": wec_range,
        }
        env.set("wec", wec_module)
        # ADDED: Register the built-in i32Matrix type with its "random" static method.
        env.set("i32Matrix", {"random": i32Matrix_random})

    def _process_python_imports(self, env):
        # Processar pytype declarations
        for decl in self.program.types:
            if isinstance(decl, PyTypeDecl):
                try:
                    ns = {}
                    exec(decl.type_def, ns)
                    env.set(decl.name, ns[decl.name])
                except Exception as e:
                    raise Exception(f"Erro ao processar pytype {decl.name}: {str(e)}")
        
        # Processar módulos Python primeiro
        modules = {}
        for decl in self.program.types:
            if isinstance(decl, PythonImport):
                try:
                    imported = __import__(decl.module)
                    modules[decl.module] = imported
                except ImportError as e:
                    raise Exception(f"Failed to import Python module {decl.module}: {str(e)}")
        
        # Registrar módulos primeiro
        for mod_name, mod in modules.items():
            env.set(mod_name, mod)
        
        # Processar pyfn declarations dentro do escopo do módulo
        for decl in self.program.types:
            if isinstance(decl, PyFnDecl):
                def make_pyfn_wrapper(fn_code, params):
                    def wrapper(*args):
                        param_bindings = {param.name: arg for param, arg in zip(params, args)}
                        try:
                            exec(fn_code, param_bindings)
                            return param_bindings.get('__return__', None)
                        except Exception as e:
                            raise Exception(f"Python function error: {str(e)}")
                    return wrapper
                
                # Registrar a função no módulo correto
                module_name = self.program.module.name if self.program.module else ""
                if module_name:
                    module_env = env.get(module_name, {})
                    pyfn = make_pyfn_wrapper(decl.code, decl.params)
                    module_env[decl.name] = pyfn
                    env.set(module_name, module_env)
                else:
                    pyfn = make_pyfn_wrapper(decl.code, decl.params)
                    env.set(decl.name, pyfn)

    def _process_module_imports(self, imported_ast, env):
        """Processa declarações Python específicas de um módulo importado"""
        # Coletar imports Python do módulo
        py_modules = {}
        for decl in imported_ast.types:
            if isinstance(decl, PythonImport):
                try:
                    py_modules[decl.module] = __import__(decl.module)
                except ImportError as e:
                    raise Exception(f"Failed to import Python module {decl.module}: {str(e)}")

        # Processar pyfn declarations do módulo
        for decl in imported_ast.types:
            if isinstance(decl, PyFnDecl):
                def make_pyfn_wrapper(fn_code, params, modules):
                    def wrapper(*args):
                        param_bindings = {param.name: arg for param, arg in zip(params, args)}
                        param_bindings.update(modules)  # Adiciona módulos importados
                        try:
                            exec(fn_code, param_bindings)
                            return param_bindings.get('__return__', None)
                        except Exception as e:
                            raise Exception(f"Python function error: {str(e)}")
                    return wrapper
                
                # Registrar no namespace do módulo
                pyfn = make_pyfn_wrapper(decl.code, decl.params, py_modules)
                env.values[decl.name] = pyfn  # Registra diretamente no dict values do Environment

        # Processar pytype declarations do módulo
        for decl in imported_ast.types:
            if isinstance(decl, PyTypeDecl):
                try:
                    ns = {'__name__': 'wec_type_module'}
                    ns.update(py_modules)
                    type_value = eval(decl.type_def, ns)
                    env.values[decl.name] = type_value  # Registra diretamente no dict values
                except Exception as e:
                    raise Exception(f"Erro ao processar pytype {decl.name}: {str(e)}")

        # Registrar o namespace do módulo
        module_name = imported_ast.module.name if imported_ast.module else os.path.splitext(os.path.basename(imported_ast.filename))[0]
        self.global_env.set(module_name, env.values)

        # Adicionar funções WEC ao namespace como ModuleFunctions
        for f in imported_ast.functions:
            if isinstance(f, FunctionDef):
                wrapped_func = ModuleFunction(f, env, self)
                env.set(f.name, wrapped_func)
        
        # Registrar o namespace completo
        self.global_env.set(module_name, env.values)

    def interpret(self, program):
        self.program = program
        if not hasattr(self.program, 'types'):
            self.program.types = []
        
        # Processar imports Python primeiro
        self._process_python_imports(self.global_env)
        
        # Then process regular imports
        if hasattr(program, "imports"):
            for import_decl in program.imports:
                self.process_import(import_decl)
        # Register type definitions, if any.
        if hasattr(program, "types"):
            for type_def in program.types:
                self.global_env.set(type_def.name, type_def)
        # ADDED: Register class definitions to support the new "new NeuralNetwork(...)" syntax.
        if hasattr(program, "classes"):
            for class_def in program.classes:
                self.global_env.set(class_def.name, class_def)
        for func in program.functions:
            self.global_env.set(func.name, func)
        # Look for the main function
        try:
            main_func = self.global_env.get("main")
        except Exception:
            raise Exception("Main function not defined.")
        result = self.call_function(main_func, [])
        # Opcional: retorna ou imprime o valor retornado pela main
        # print("Resultado da main:", result)

    def call_function(self, func, args):
        if isinstance(func, ModuleFunction):
            # Usar o ambiente do módulo como escopo externo
            local_env = Environment(outer=func.module_env)
            func_def = func.func_def
        else:
            local_env = Environment(outer=self.global_env)
            func_def = func
        
        for param, arg in zip(func_def.params, args):
            local_env.set(param.name, arg)
        
        try:
            return self.execute_block(func_def.body, local_env)
        except ReturnException as ret:
            return ret.value

    def execute_block(self, block, env):
        for stmt in block.statements:
            self.execute(stmt, env)
    def execute(self, stmt, env):
        if isinstance(stmt, VarDecl):
            if stmt.initializer is None:
                if stmt.type.endswith("List"):
                    value = WECList()
                    value.wec_type = stmt.type
                elif stmt.type.endswith("Matrix"):   # ADDED: create a matrix instance if type endswith "Matrix"
                    value = WECMatrix()
                    value.wec_type = stmt.type
                else:
                    value = None
            else:
                value = self.evaluate(stmt.initializer, env)
            if value is not None and not type_matches(stmt.type, get_type(value)):
                raise Exception(f"Type error in variable '{stmt.name}': declared {stmt.type}, but got {get_type(value)}")
            env.set(stmt.name, value)
        elif isinstance(stmt, ReturnStmt):
            value = self.evaluate(stmt.expression, env)
            raise ReturnException(value)
        elif isinstance(stmt, ExprStmt):
            self.evaluate(stmt.expression, env)
        elif isinstance(stmt, IfStmt):
            if self.evaluate(stmt.condition, env):
                self.execute_block(stmt.then_block, env)
            else:
                executed = False
                for (cond, block) in stmt.elif_blocks:
                    if self.evaluate(cond, env):
                        self.execute_block(block, env)
                        executed = True
                        break
                if not executed and stmt.else_block is not None:
                    # Se o else_block é um Block, usa execute_block; se for um IfStmt, usa execute.
                    if isinstance(stmt.else_block, Block):
                        self.execute_block(stmt.else_block, env)
                    else:
                        self.execute(stmt.else_block, env)
        elif isinstance(stmt, WhileStmt):
            return self.execute_while(stmt, env)
        elif isinstance(stmt, ForStmt):
            return self.execute_for(stmt, env)
        elif isinstance(stmt, Assignment):
            # Execute the assignment by evaluating the right-hand side.
            if hasattr(stmt.target, "index") and stmt.target.__class__.__name__ == "IndexExpr":
                array_obj = self.evaluate(stmt.target.receiver, env)
                idx = self.evaluate(stmt.target.index, env)
                array_obj[idx] = self.evaluate(stmt.value, env)
            elif isinstance(stmt.target, MemberAccess):
                val = self.evaluate(stmt.value, env)
                receiver = self.evaluate(stmt.target.receiver, env)
                if isinstance(receiver, dict):
                    receiver[stmt.target.member] = val
                elif isinstance(receiver, Instance):
                    receiver.fields[stmt.target.member] = val
                else:
                    setattr(receiver, stmt.target.member, val)
            elif isinstance(stmt.target, Identifier):
                value = self.evaluate(stmt.value, env)
                env.set(stmt.target.name, value)
            else:
                raise Exception("Invalid assignment target")
        else:
            raise Exception("Unknown statement type: " + str(stmt))
    def execute_while(self, stmt, env):
        while self.evaluate(stmt.condition, env):
            self.execute_block(stmt.body, env)
    def execute_for(self, stmt, env):
        # Evaluate the range expression
        range_expr = self.evaluate(stmt.range_expr, env)
        
        # Iterate through the range
        for value in range_expr:
            # Create a new environment for the loop iteration
            local_env = Environment(outer=env)
            local_env.set(stmt.var_name, value)
            self.execute_block(stmt.body, local_env)
    def evaluate(self, expr, env):
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Identifier):
            return env.get(expr.name)
        elif isinstance(expr, ListExpr):
            return [self.evaluate(el, env) for el in expr.elements]
        elif isinstance(expr, UnaryOp):
            value = self.evaluate(expr.value, env)
            if expr.op == '-':
                return -value
            else:
                raise Exception("Unknown unary operator: " + expr.op)
        elif isinstance(expr, MemberAccess):
            return self.evaluate_member_access(expr, env)
        elif isinstance(expr, CallExpr):
            return self.evaluate_call_expr(expr, env)
        elif isinstance(expr, BinOp):
            left = self.evaluate(expr.left, env)
            right = self.evaluate(expr.right, env)
            op = expr.op
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right
            elif op == '%':
                return left % right
            elif op == '<=':
                return left <= right
            elif op == '<':
                return left < right
            elif op == '>=':
                return left >= right
            elif op == '>':
                return left > right
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right
            else:
                raise Exception("Unknown binary operator: " + op)
        elif isinstance(expr, RecordLiteral):
            record = {}
            for key, field_expr in expr.fields.items():
                record[key] = self.evaluate(field_expr, env)
            record["wec_type"] = expr.type_name
            # If the record's type has an implementation, attach its methods.
            try:
                type_def = env.get(expr.type_name)
                if hasattr(type_def, "methods"):
                    for method in type_def.methods:
                        record[method.name] = method
            except Exception:
                pass
            return record
        elif isinstance(expr, NewExpr):
            return self.evaluate_new_expr(expr, env)
        elif expr.__class__.__name__ == "IndexExpr":
            array_obj = self.evaluate(expr.receiver, env)
            index = self.evaluate(expr.index, env)
            return array_obj[index]
        elif isinstance(expr, FunctionDef) or hasattr(expr, 'params'):
            # Wrap the function definition so that it becomes a Python callable.
            return lambda *args: self.call_function(expr, args)
        else:
            raise Exception("Unknown expression: " + str(expr))

    def evaluate_new_expr(self, node, env):
        # If the new expression node has an extra attribute 'member', combine it with type_name.
        if hasattr(node, "member") and node.member:
            node.type_name = f"{node.type_name}::{node.member}"

        # Now check if the (possibly modified) type name is qualified.
        if "::" in node.type_name:
            module_name, type_name = node.type_name.split("::", 1)
            module_def = env.get(module_name)
            if not isinstance(module_def, dict):
                raise Exception(f"Module '{module_name}' not found or not a module.")
            type_def = module_def.get(type_name)
            if type_def is None:
                raise Exception(f"Type '{type_name}' not found in module '{module_name}'.")
        else:
            type_def = env.get(node.type_name)
            if isinstance(type_def, dict):
                raise Exception(
                    f"Cannot instantiate module '{node.type_name}'. "
                    f"Use member access (e.g., '{node.type_name}::TypeName') to instantiate a type from the module."
                )

        # Special handling for built-in types (like File)
        if hasattr(type_def, "builtin_file") and type_def.builtin_file:
            args = []
            if hasattr(node, "args") and node.args:
                args = [self.evaluate(arg.value, env) for arg in node.args]
            elif hasattr(node, "arguments") and node.arguments:
                args = [self.evaluate(arg.value, env) for arg in node.arguments]
            elif hasattr(node, "arg_list") and node.arg_list:
                args = [self.evaluate(arg.value, env) for arg in node.arg_list]
            if len(args) not in (0, 2):
                raise Exception("File constructor requires either 0 or 2 arguments: filename and mode")
            if len(args) == 0:
                return type_def.new_instance()
            else:
                return type_def.new_instance(*args)

        # Fallback to normal interpreter instantiation for other types.
        instance = Instance(type_def)
        # If using a qualified type name, update the instance type accordingly.
        if "::" in node.type_name:
            instance.wec_type = node.type_name

        # Initialize public fields from the class definition.
        for member in type_def.members:
            if isinstance(member, tuple) and member[0] == "field":
                # member is ("field", fieldName, fieldType)
                instance.fields[member[1]] = None
        # Look for a constructor method (named "constructor") in the definition.
        constructor = instance.get_method("constructor")
        # Only call the constructor immediately if arguments were provided.
        if constructor is not None and node.args:
            local_env = Environment(outer=env)
            local_env.set("this", instance)
            if len(constructor.params) != len(node.args):
                raise Exception("Número de argumentos incompatível no construtor")
            for param, arg in zip(constructor.params, node.args):
                local_env.set(param.name, self.evaluate(arg.value, env))
            result = self.execute_block(constructor.body, local_env)
            instance.fields["call"] = result
        else:
            instance.fields["call"] = None
        return instance

    def instantiate_class(self, class_def, args):
        # Create a new instance (dictionary) with the class name as its type.
        instance = {"wec_type": class_def.name}
        # Process class members: install fields (defaulting to None) and methods.
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "field":
                # member is ("field", fieldName, fieldType)
                field_name = member[1]
                instance[field_name] = None
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "method":
                method_def = member[1]
                # Bind method to instance: create a closure that sets 'this' to instance.
                def bound_method(*m_args, method_def=method_def):
                    local_env = Environment(outer=self.global_env)
                    local_env.set("this", instance)
                    if len(method_def.params) != len(m_args):
                        raise Exception("Incorrect number of arguments in method call: " + method_def.name)
                    for param, arg in zip(method_def.params, m_args):
                        local_env.set(param.name, arg)
                    try:
                        self.execute_block(method_def.body, local_env)
                    except ReturnException as ret:
                        return ret.value
                # Attach the bound method to the instance.
                instance[method_def.name] = bound_method
        # Look for a constructor method (named "constructor") if present.
        for member in class_def.members:
            if isinstance(member, tuple) and member[0] == "method" and member[1].name == "constructor":
                constructor = member[1]
                local_env = Environment(outer=self.global_env)
                local_env.set("this", instance)
                if len(constructor.params) != len(args):
                    raise Exception("Incorrect number of arguments in constructor of " + class_def.name)
                for param, arg in zip(constructor.params, args):
                    local_env.set(param.name, arg)
                try:
                    self.execute_block(constructor.body, local_env)
                except ReturnException as ret:
                    pass
                break
        return instance

    def evaluate_member_access(self, expr, env):
        receiver = self.evaluate(expr.receiver, env)
        # Se o receptor for uma instância (do tipo Instance), busque primeiro nos campos.
        if isinstance(receiver, Instance):
            if expr.member in receiver.fields:
                return receiver.fields[expr.member]
            method = receiver.get_method(expr.member)
            if method is not None:
                return lambda *args: self.call_method(receiver, method, args, env)
            raise Exception(f"Field or method '{expr.member}' not found in instance of {receiver.class_def.name}")

        # Se o receptor for um EnumDef, procure na lista de variantes.
        if isinstance(receiver, EnumDef):
            if hasattr(receiver, "variants") and (expr.member in receiver.variants):
                return EnumValue(receiver, expr.member)
            else:
                raise Exception(f"Enum variant '{expr.member}' not found in enum {receiver.name}")

        # Se o receptor for um dicionário (por exemplo, um módulo ou built-in registrado), faça lookup no dicionário.
        if isinstance(receiver, dict):
            member = receiver.get(expr.member)
            if member is None:
                raise Exception(f"Member '{expr.member}' not found in {receiver}")
            return member

        # Se o receptor for uma primitiva (int, float, str), trate métodos especiais.
        if isinstance(receiver, (int, float, str)):
            if expr.member == "to_int":
                return lambda *args: int(receiver)
            elif expr.member == "to_str":
                return lambda *args: str(receiver)
            elif expr.member == "to_float":
                return lambda *args: float(receiver)
            else:
                raise Exception(f"Member '{expr.member}' not found in primitive type {type(receiver).__name__}")

        # Caso contrário, caia no lookup padrão usando getattr.
        try:
            return getattr(receiver, expr.member)
        except AttributeError:
            raise Exception(f"Member '{expr.member}' not found in object {receiver}")
    
    def call_method(self, instance, method, args, env):
        local_env = Environment(outer=env)
        local_env.set("this", instance)
        # Inject module-level definitions if the instance's type is qualified (e.g., "math::Math")
        if "::" in instance.wec_type:
            module_name, _ = instance.wec_type.split("::", 1)
            try:
                module_def = self.global_env.get(module_name)
            except Exception:
                module_def = None
            if module_def and isinstance(module_def, dict):
                for key, value in module_def.items():
                    # Avoid overwriting existing definitions in the local environment.
                    if key not in local_env.values:
                        local_env.set(key, value)
        for param, arg in zip(method.params, args):
            local_env.set(param.name, arg)
        try:
            return self.execute_block(method.body, local_env)
        except ReturnException as ret:
            return ret.value

    def evaluate_call_expr(self, node, env):
        callee = self.evaluate(node.callee, env)
        args = [self.evaluate(arg.value, env) for arg in node.args]
        if isinstance(callee, Instance):
            # If an instance is called, use its 'constructor' method (if defined) as its call operator.
            method = callee.get_method("constructor")
            if method is not None:
                return self.call_method(callee, method, args, env)
            else:
                raise Exception("Instance not callable: no constructor method found")
        elif isinstance(callee, FunctionDef):
            return self.call_function(callee, args)
        elif callable(callee):
            return callee(*args)
        else:
            raise Exception("Attempt to call a non-function: " + str(callee))

    def process_import(self, import_decl):
        if isinstance(import_decl, PythonImport):
            return
        
        try:
            filename = import_decl.filename
            
            # Determinar o caminho base
            if filename.startswith("./"):
                # Usar diretório do arquivo sendo executado
                if len(sys.argv) > 1 and sys.argv[1].endswith('.wec'):
                    base_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
                else:
                    base_dir = os.getcwd()
                full_path = os.path.join(base_dir, filename[2:])
            else:
                # Usar diretório do usuário/weclibs
                home_dir = os.path.expanduser("~")
                full_path = os.path.join(home_dir, "weclibs", filename)
            
            # Adicionar extensão .wec se necessário
            if not full_path.endswith(".wec"):
                full_path += ".wec"
            
            # Verificar se o arquivo existe
            if not os.path.exists(full_path):
                raise Exception(f"Arquivo importado não encontrado: {full_path}")
            
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            imported_ast = parse_wec(source)
            
            # Criar namespace para o módulo importado
            module_env = Environment()
            
            # Processar declarações Python do módulo importado
            self._process_module_imports(imported_ast, module_env)
            
            # Adicionar funções WEC e tipos ao namespace
            for f in imported_ast.functions:
                if isinstance(f, FunctionDef):
                    wrapped_func = ModuleFunction(f, module_env, self)
                    module_env.set(f.name, wrapped_func)
            
            # Adicionar tipos (incluindo classes) ao namespace
            for t in imported_ast.types:
                if isinstance(t, (StructDef, EnumDef, JoinDef, ClassDef, PyTypeDecl)):
                    module_env.set(t.name, t)  # Registra tipos diretamente no module_env
            
            # Registrar o namespace completo
            module_name = imported_ast.module.name if imported_ast.module else os.path.splitext(os.path.basename(filename))[0]
            self.global_env.set(module_name, module_env.values)
            
        except Exception as e:
            raise Exception(f"Erro importando {import_decl.filename}: {str(e)}")

# Nova classe para representar instâncias de classes
class Instance:
    def __init__(self, class_def):
        self.class_def = class_def  # A definição da classe (ex: ClassDef)
        self.fields = {}  # Campos armazenados na instância
        self.wec_type = class_def.name  # Define the type name as the class name

    def get_method(self, name):
        # Procura por um método com o nome solicitado na definição da classe.
        for member in self.class_def.members:
            # Cada membro de método vem na forma ("method", FunctionDef)
            if isinstance(member, tuple) and member[0] == "method":
                method = member[1]
                if method.name == name:
                    return method
        return None

class WECList(list):
    def length(self):
        return len(self)

    def foreach(self, func):
        for item in self:
            func(item)

    def map(self, func):
        newList = WECList()
        newList.wec_type = self.wec_type
        for item in self:
            newList.append(func(item))
        return newList

    def filter(self, func):
        newList = WECList()
        newList.wec_type = self.wec_type
        for item in self:
            if func(item):
                newList.append(item)
        return newList

class WECMatrix(list):
    def to(self, dims):
        # Set the number of dimensions (allow between 1 and 10)
        if not (1 <= dims <= 10):
            raise Exception("Matrix dimensions must be between 1 and 10")
        self.dimensions = dims
        return self  # allow chaining if desired

    def append(self, value):
        if hasattr(self, "dimensions") and self.dimensions and isinstance(value, list):
            if len(value) > 0 and any(isinstance(x, list) for x in value):
                for row in value:
                    if not isinstance(row, WECList):
                        new_row = WECList(row)
                        new_row.wec_type = self.wec_type.replace("Matrix", "List") if hasattr(self, "wec_type") else "List"
                        super(WECMatrix, self).append(new_row)
                    else:
                        super(WECMatrix, self).append(row)
                return
            else:
                new_row = WECList(value)
                new_row.wec_type = self.wec_type.replace("Matrix", "List") if hasattr(self, "wec_type") else "List"
                super(WECMatrix, self).append(new_row)
                return
        super(WECMatrix, self).append(value)

    def dot(self, other):
        # NEW: If compiled mode is enabled, use NumPy for the dot operation.
        from interpreter import COMPILED_MODE
        if COMPILED_MODE:
            import numpy as np
            try:
                arr1 = np.array([list(row) for row in self], dtype=float)
                arr2 = np.array([list(row) for row in other], dtype=float)
            except Exception as e:
                raise Exception("Error converting matrix to numpy array: " + str(e))
            result_arr = np.dot(arr1, arr2)
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for row in result_arr:
                new_row = WECList(row.tolist())
                new_row.wec_type = "i32List"
                result.append(new_row)
            return result
        # Fallback to original implementation
        if not self or not other:
            raise Exception("Empty matrix cannot be used for dot multiplication")
        r = len(self)
        c = len(self[0])
        for row in self:
            if len(row) != c:
                raise Exception("Inconsistent dimensions in first matrix")
        if len(other) != c:
            raise Exception("Incompatible dimensions for matrix multiplication")
        k = len(other[0])
        result = WECMatrix()
        result.to(self.dimensions)   # assume uma matriz 2D
        result.wec_type = self.wec_type
        for i in range(r):
            new_row = WECList()
            new_row.wec_type = "i32List"
            for j in range(k):
                sum_val = 0
                for p in range(c):
                    sum_val += self[i][p] * other[p][j]
                new_row.append(sum_val)
            result.append(new_row)
        return result

    def transpose(self):
        # Retorna a transposta da matriz.
        if not self:
            return WECMatrix()
        r = len(self)
        c = len(self[0])
        result = WECMatrix()
        result.to(self.dimensions)
        result.wec_type = self.wec_type
        for j in range(c):
            new_row = WECList()
            new_row.wec_type = "i32List"
            for i in range(r):
                new_row.append(self[i][j])
            result.append(new_row)
        return result

    def __add__(self, other):
        from interpreter import COMPILED_MODE
        if COMPILED_MODE:
            import numpy as np
            if isinstance(other, (int, float)):
                arr = np.array([list(row) for row in self], dtype=float) + other
            elif isinstance(other, WECMatrix):
                arr_self = np.array([list(row) for row in self], dtype=float)
                arr_other = np.array([list(row) for row in other], dtype=float)
                arr = arr_self + arr_other
            else:
                raise Exception("Addition not supported between matrix and non-matrix")
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            if arr.ndim == 1:
                arr = np.array([arr])
            for row in arr:
                new_row = WECList(row.tolist())
                new_row.wec_type = "i32List"
                result.append(new_row)
            return result
        # Original implementation for scalar addition or matrix addition with broadcasting.
        if isinstance(other, (int, float)):
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for row in self:
                new_row = WECList()
                new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                for element in row:
                    new_row.append(element + other)
                result.append(new_row)
            return result
        elif isinstance(other, WECMatrix):
            rows_self = len(self)
            cols_self = len(self[0]) if rows_self > 0 else 0
            rows_other = len(other)
            cols_other = len(other[0]) if rows_other > 0 else 0
            if rows_self == rows_other:
                result_rows = rows_self
            elif rows_self == 1:
                result_rows = rows_other
            elif rows_other == 1:
                result_rows = rows_self
            else:
                raise Exception("Incompatible row dimensions for matrix addition")
            if cols_self == cols_other:
                result_cols = cols_self
            elif cols_self == 1:
                result_cols = cols_other
            elif cols_other == 1:
                result_cols = cols_self
            else:
                raise Exception("Incompatible column dimensions for matrix addition")
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for i in range(result_rows):
                new_row = WECList()
                new_row.wec_type = "i32List"
                for j in range(result_cols):
                    if rows_self == 1:
                        val_self = self[0][0] if cols_self == 1 else self[0][j]
                    else:
                        val_self = self[i][0] if cols_self == 1 else self[i][j]
                    if rows_other == 1:
                        val_other = other[0][0] if cols_other == 1 else other[0][j]
                    else:
                        val_other = other[i][0] if cols_other == 1 else other[i][j]
                    new_row.append(val_self + val_other)
                result.append(new_row)
            return result
        else:
            raise Exception("Addition not supported between matrix and non-matrix")

    def __radd__(self, other):
        # Se o operando à esquerda é um escalar, use a definição de __add__
        if isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise Exception("Addition not supported for these operands")

    def __sub__(self, other):
        # Se other é um escalar, faz a subtração elemento a elemento.
        if isinstance(other, (int, float)):
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for row in self:
                new_row = WECList()
                new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                for element in row:
                    new_row.append(element - other)
                result.append(new_row)
            return result
        elif isinstance(other, WECMatrix):
            rows_self = len(self)
            cols_self = len(self[0]) if rows_self > 0 else 0
            rows_other = len(other)
            cols_other = len(other[0]) if rows_other > 0 else 0

            if rows_self == rows_other:
                result_rows = rows_self
            elif rows_self == 1:
                result_rows = rows_other
            elif rows_other == 1:
                result_rows = rows_self
            else:
                raise Exception("Incompatible row dimensions for matrix subtraction")

            if cols_self == cols_other:
                result_cols = cols_self
            elif cols_self == 1:
                result_cols = cols_other
            elif cols_other == 1:
                result_cols = cols_self
            else:
                raise Exception("Incompatible column dimensions for matrix subtraction")

            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for i in range(result_rows):
                new_row = WECList()
                new_row.wec_type = "i32List"
                for j in range(result_cols):
                    if rows_self == 1:
                        val_self = self[0][0] if cols_self == 1 else self[0][j]
                    else:
                        val_self = self[i][0] if cols_self == 1 else self[i][j]

                    if rows_other == 1:
                        val_other = other[0][0] if cols_other == 1 else other[0][j]
                    else:
                        val_other = other[i][0] if cols_other == 1 else other[i][j]

                    new_row.append(val_self - val_other)
                result.append(new_row)
            return result
        else:
            raise Exception("Subtraction not supported between matrix and non-matrix")

    def __rsub__(self, other):
        # Se o operando à esquerda é um escalar, computa escalar - matrix elemento a elemento.
        if isinstance(other, (int, float)):
            result = WECMatrix()
            result.to(self.dimensions)
            result.wec_type = self.wec_type
            for row in self:
                new_row = WECList()
                new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                for element in row:
                    new_row.append(other - element)
                result.append(new_row)
            return result
        else:
            raise Exception("Subtraction not supported for these operands")

    def __neg__(self):
        # Retorna uma nova matriz com cada elemento negado
        result = WECMatrix()
        result.to(self.dimensions)
        result.wec_type = self.wec_type
        for row in self:
            new_row = WECList()
            new_row.wec_type = "i32List"
            for element in row:
                new_row.append(-element)
            result.append(new_row)
        return result

    def __mul__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(element * other)
                 result.append(new_row)
             return result
         elif isinstance(other, WECMatrix):
             # Elementwise multiplication com broadcasting
             rows_self = len(self)
             cols_self = len(self[0]) if rows_self > 0 else 0
             rows_other = len(other)
             cols_other = len(other[0]) if rows_other > 0 else 0
 
             if rows_self == rows_other:
                 result_rows = rows_self
             elif rows_self == 1:
                 result_rows = rows_other
             elif rows_other == 1:
                 result_rows = rows_self
             else:
                 raise Exception("Incompatible row dimensions for matrix multiplication")
 
             if cols_self == cols_other:
                 result_cols = cols_self
             elif cols_self == 1:
                 result_cols = cols_other
             elif cols_other == 1:
                 result_cols = cols_self
             else:
                 raise Exception("Incompatible column dimensions for matrix multiplication")
 
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for i in range(result_rows):
                 new_row = WECList()
                 new_row.wec_type = "i32List"
                 for j in range(result_cols):
                     if rows_self == 1:
                         val_self = self[0][0] if cols_self == 1 else self[0][j]
                     else:
                         val_self = self[i][0] if cols_self == 1 else self[i][j]
                     if rows_other == 1:
                         val_other = other[0][0] if cols_other == 1 else other[0][j]
                     else:
                         val_other = other[i][0] if cols_other == 1 else other[i][j]
                     new_row.append(val_self * val_other)
                 result.append(new_row)
             return result
         else:
             raise Exception("Multiplication not supported between matrix and non-matrix")

    def __rmul__(self, other):
         if isinstance(other, (int, float)):
             return self.__mul__(other)
         else:
             raise Exception("Multiplication not supported for these operands")

    def __truediv__(self, other):
         if isinstance(other, (int, float)):
             if other == 0:
                 raise Exception("Division by zero")
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(element / other)
                 result.append(new_row)
             return result
         elif isinstance(other, WECMatrix):
             # Elementwise division com broadcasting
             rows_self = len(self)
             cols_self = len(self[0]) if rows_self > 0 else 0
             rows_other = len(other)
             cols_other = len(other[0]) if rows_other > 0 else 0
 
             if rows_self == rows_other:
                 result_rows = rows_self
             elif rows_self == 1:
                 result_rows = rows_other
             elif rows_other == 1:
                 result_rows = rows_self
             else:
                 raise Exception("Incompatible row dimensions for matrix division")
 
             if cols_self == cols_other:
                 result_cols = cols_self
             elif cols_self == 1:
                 result_cols = cols_other
             elif cols_other == 1:
                 result_cols = cols_self
             else:
                 raise Exception("Incompatible column dimensions for matrix division")
 
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for i in range(result_rows):
                 new_row = WECList()
                 new_row.wec_type = "i32List"
                 for j in range(result_cols):
                     if rows_self == 1:
                         val_self = self[0][0] if cols_self == 1 else self[0][j]
                     else:
                         val_self = self[i][0] if cols_self == 1 else self[i][j]
                     if rows_other == 1:
                         val_other = other[0][0] if cols_other == 1 else other[0][j]
                     else:
                         val_other = other[i][0] if cols_other == 1 else other[i][j]
                     if val_other == 0:
                         raise Exception("Division by zero in matrix division")
                     new_row.append(val_self / val_other)
                 result.append(new_row)
             return result
         else:
             raise Exception("Division not supported between matrix and non-matrix")

    def __rtruediv__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     if element == 0:
                         raise Exception("Division by zero in matrix division")
                     new_row.append(other / element)
                 result.append(new_row)
             return result
         else:
             raise Exception("Division not supported for these operands")

    def __mod__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(element % other)
                 result.append(new_row)
             return result
         elif isinstance(other, WECMatrix):
             rows_self = len(self)
             cols_self = len(self[0]) if rows_self > 0 else 0
             rows_other = len(other)
             cols_other = len(other[0]) if rows_other > 0 else 0
 
             if rows_self == rows_other:
                 result_rows = rows_self
             elif rows_self == 1:
                 result_rows = rows_other
             elif rows_other == 1:
                 result_rows = rows_self
             else:
                 raise Exception("Incompatible row dimensions for matrix modulo")
 
             if cols_self == cols_other:
                 result_cols = cols_self
             elif cols_self == 1:
                 result_cols = cols_other
             elif cols_other == 1:
                 result_cols = cols_self
             else:
                 raise Exception("Incompatible column dimensions for matrix modulo")
 
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for i in range(result_rows):
                 new_row = WECList()
                 new_row.wec_type = "i32List"
                 for j in range(result_cols):
                     if rows_self == 1:
                         val_self = self[0][0] if cols_self == 1 else self[0][j]
                     else:
                         val_self = self[i][0] if cols_self == 1 else self[i][j]
                     if rows_other == 1:
                         val_other = other[0][0] if cols_other == 1 else other[0][j]
                     else:
                         val_other = other[i][0] if cols_other == 1 else other[i][j]
                     new_row.append(val_self % val_other)
                 result.append(new_row)
             return result
         else:
             raise Exception("Modulo not supported between matrix and non-matrix")

    def __rmod__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(other % element)
                 result.append(new_row)
             return result
         else:
             raise Exception("Modulo not supported for these operands")

    def __pow__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(element ** other)
                 result.append(new_row)
             return result
         elif isinstance(other, WECMatrix):
             rows_self = len(self)
             cols_self = len(self[0]) if rows_self > 0 else 0
             rows_other = len(other)
             cols_other = len(other[0]) if rows_other > 0 else 0
 
             if rows_self == rows_other:
                 result_rows = rows_self
             elif rows_self == 1:
                 result_rows = rows_other
             elif rows_other == 1:
                 result_rows = rows_self
             else:
                 raise Exception("Incompatible row dimensions for matrix exponentiation")
 
             if cols_self == cols_other:
                 result_cols = cols_self
             elif cols_self == 1:
                 result_cols = cols_other
             elif cols_other == 1:
                 result_cols = cols_self
             else:
                 raise Exception("Incompatible column dimensions for matrix exponentiation")
 
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for i in range(result_rows):
                 new_row = WECList()
                 new_row.wec_type = "i32List"
                 for j in range(result_cols):
                     if rows_self == 1:
                         val_self = self[0][0] if cols_self == 1 else self[0][j]
                     else:
                         val_self = self[i][0] if cols_self == 1 else self[i][j]
                     if rows_other == 1:
                         val_other = other[0][0] if cols_other == 1 else other[0][j]
                     else:
                         val_other = other[i][0] if cols_other == 1 else other[i][j]
                     new_row.append(val_self ** val_other)
                 result.append(new_row)
             return result
         else:
             raise Exception("Exponentiation not supported between matrix and non-matrix")

    def __rpow__(self, other):
         if isinstance(other, (int, float)):
             result = WECMatrix()
             result.to(self.dimensions)
             result.wec_type = self.wec_type
             for row in self:
                 new_row = WECList()
                 new_row.wec_type = row.wec_type if hasattr(row, "wec_type") else "i32List"
                 for element in row:
                     new_row.append(other ** element)
                 result.append(new_row)
             return result
         else:
             raise Exception("Exponentiation not supported for these operands")

def i32Matrix_random(rows, cols):
    import random
    m = WECMatrix()
    m.to(2)
    m.wec_type = "i32Matrix"
    for i in range(rows):
        row = WECList()
        row.wec_type = "i32List"
        for j in range(cols):
            row.append(random.randint(0, 9))
        m.append(row)
    return m

class ModuleFunction:
    def __init__(self, func_def, module_env, interpreter):
        self.func_def = func_def
        self.module_env = module_env
        self.interpreter = interpreter

    def __call__(self, *args):
        local_env = Environment(outer=self.module_env)
        for param, arg in zip(self.func_def.params, args):
            local_env.set(param.name, arg)
        try:
            return self.interpreter.execute_block(self.func_def.body, local_env)
        except ReturnException as ret:
            return ret.value
